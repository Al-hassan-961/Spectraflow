/*
 * Spectraflow - ESP32 Wi-Fi Channel State Information (CSI) collector.
 *
 * This file contains:
 *   1. The reusable collector component (CSI capture, UDP framing/streaming and
 *      the raw-802.11-frame illuminator).
 *   2. The default example application entry point (app_main) at the bottom of
 *      the file, plus the USER CONFIGURATION block below.
 *
 * If you integrate the collector into a larger application, keep everything down
 * to the "APPLICATION ENTRY POINT" banner and delete the app_main section.
 *
 * Threading model
 * ---------------
 *   - csi_rx_callback() runs in the Wi-Fi driver task context. It performs one
 *     memcpy into a pre-allocated queue item and nothing else: no allocation, no
 *     logging, no socket or name-service I/O, and it never blocks.
 *   - csi_sender_task() is a lower-priority FreeRTOS task that pops queue items,
 *     serializes the frozen 24-byte header plus payload into a reusable stack
 *     buffer and hands it to sendto() on a persistent UDP socket.
 *   - csi_illuminator_task() is only used by the transmitter role and emits raw
 *     802.11 frames at the configured rate.
 *
 * The CSI callback is invoked from a task, not from an ISR, so the queue is fed
 * with the plain xQueueSend() variant and a zero block time. The FromISR()
 * variants must not be used outside an interrupt context.
 */

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_bit_defs.h" /* BIT0 / BIT1 */
#include "esp_event.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "csi_collector.h"

/* ========================================================================== */
/* ==== USER CONFIGURATION - EDIT THIS BLOCK BEFORE FLASHING ================ */
/* ========================================================================== */
/*
 * Every node runs the same binary. A node becomes a transmitter or a receiver
 * purely through CSI_NODE_ROLE (plus the network settings below).
 *
 * Receiver node : CSI_NODE_ROLE = CSI_ROLE_RECEIVER, plus your AP credentials
 *                 and the IPv4 address of the host running Spectraflow.
 * Transmitter   : CSI_NODE_ROLE = CSI_ROLE_TRANSMITTER. The transmitter never
 *                 associates and never uploads, so the destination address, the
 *                 destination port and the AP credentials are ignored.
 */

/* Role of this node: CSI_ROLE_RECEIVER or CSI_ROLE_TRANSMITTER. */
#define CSI_NODE_ROLE CSI_ROLE_RECEIVER

/* Unique per-node identifier reported in header bytes 4..5. */
#define CSI_NODE_ID 1

/* UDP destination: the host that runs Spectraflow. Receiver role only. */
#define CSI_DEST_IP "192.168.1.100"
#define CSI_DEST_PORT 5005

/*
 * Wi-Fi channel used for sensing. The transmitter, the receiver and the access
 * point the receiver associates with must all be on this channel, otherwise the
 * receiver measures ambient traffic instead of the illuminator.
 */
#define CSI_WIFI_CHANNEL 6

/* WIFI_BW_HT20 -> up to 64 subcarriers, WIFI_BW_HT40 -> up to 128 subcarriers. */
#define CSI_WIFI_BANDWIDTH WIFI_BW_HT20

/*
 * Receiver role: credentials of the access point the receiver connects to (the
 * host must be reachable on that LAN). Leave the SSID empty to run unassociated
 * on CSI_WIFI_CHANNEL: the node then captures CSI but cannot upload it because it
 * has no IP address.
 */
#define CSI_WIFI_SSID "YOUR_AP_SSID"
#define CSI_WIFI_PASSWORD "YOUR_AP_PASSWORD"

/* Transmitter role: illuminator frame rate in frames per second (default 100). */
#define CSI_TX_FRAMES_PER_SEC 100

/*
 * Transmitter role: PHY rate of the illuminator frames. A legacy OFDM rate keeps
 * the CSI deterministic and yields LLTF-based CSI. WIFI_PHY_RATE_MCS0_LGI can be
 * used to probe HT-LTF CSI on targets that accept an HT rate here.
 */
#define CSI_TX_PHY_RATE WIFI_PHY_RATE_6M

/*
 * Optional receiver-side transmitter filter. When enabled, only CSI of frames
 * whose transmitter address equals CSI_MAC_FILTER is captured. The illuminator
 * logs its own MAC address at boot; put that value here to reject ambient
 * traffic. Disabled by default.
 */
#define CSI_MAC_FILTER_EN 0
#define CSI_MAC_FILTER {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01}

/* How often app_main prints the counters (milliseconds). */
#define CSI_STATS_INTERVAL_MS 5000

/* ========================================================================== */
/* ==== END OF USER CONFIGURATION =========================================== */
/* ========================================================================== */

static const char *TAG = "csi_collector";

/* -------------------------------------------------------------------------- */
/* Compile-time tuning                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Hand-off queue depth. Each slot holds one full CSI buffer, so this costs about
 * CSI_QUEUE_DEPTH * (sizeof(csi_frame_msg_t) + queue overhead) bytes of heap
 * (~4.4 KB with the default depth of 16).
 */
#ifndef CSI_QUEUE_DEPTH
#define CSI_QUEUE_DEPTH 16
#endif

/** Stack and priority of the framing/upload task (below the Wi-Fi task). */
#ifndef CSI_SENDER_TASK_STACK
#define CSI_SENDER_TASK_STACK 4096
#endif
#ifndef CSI_SENDER_TASK_PRIO
#define CSI_SENDER_TASK_PRIO 4
#endif

/** Stack and priority of the transmitter illuminator task. */
#ifndef CSI_ILLUMINATOR_TASK_STACK
#define CSI_ILLUMINATOR_TASK_STACK 3072
#endif
#ifndef CSI_ILLUMINATOR_TASK_PRIO
#define CSI_ILLUMINATOR_TASK_PRIO 5
#endif

/** How long csi_collector_stop() waits for a worker task to leave. */
#ifndef CSI_TASK_STOP_TIMEOUT_MS
#define CSI_TASK_STOP_TIMEOUT_MS 1000
#endif

/**
 * ESP32 hardware quirk: some silicon revisions return a stale final 32-bit word
 * in the CSI buffer. The driver exposes first_word_invalid but has no matching
 * flag for the tail, so the tail flag is raised when either this constant is set
 * or the delivered length is not a multiple of 4 bytes (an incomplete final
 * word). Define it globally (for example with a build flag) if your target shows
 * the quirk.
 */
#ifndef CSI_ASSUME_LAST_WORD_INVALID
#define CSI_ASSUME_LAST_WORD_INVALID 0
#endif

/**
 * esp_wifi_80211_tx() expects the raw MAC frame *without* the 4-byte FCS: its
 * length contract starts at 24 bytes (a bare MAC header), and the MAC hardware
 * computes and appends the FCS on air. The collector therefore keeps the FCS out
 * of the transmitted buffer by default. csi_fcs_crc32() below is nevertheless a
 * complete, self-tested 802.11 FCS implementation; set CSI_TX_APPEND_FCS to 1
 * only if a driver revision is verified to require the FCS inside the buffer.
 */
#ifndef CSI_TX_APPEND_FCS
#define CSI_TX_APPEND_FCS 0
#endif

/* Illuminator frame geometry: 24-byte 802.11 MAC header + LLC/SNAP + marker payload. */
#define CSI_TX_MAC_HEADER_BYTES 24 /* 802.11 MAC header; unrelated to the CSI wire header */
#define CSI_TX_LLC_SNAP_BYTES 8
#define CSI_TX_MARKER_BYTES 8
#define CSI_TX_COUNTER_BYTES 4
#define CSI_TX_PAYLOAD_BYTES (CSI_TX_MARKER_BYTES + CSI_TX_COUNTER_BYTES + 4)
#define CSI_TX_FRAME_BYTES (CSI_TX_MAC_HEADER_BYTES + CSI_TX_LLC_SNAP_BYTES + CSI_TX_PAYLOAD_BYTES)
#define CSI_TX_FRAME_MAX_BYTES (CSI_TX_FRAME_BYTES + 4) /* room for an optional FCS */

/*
 * Wi-Fi 6 (HE) targets redefine wifi_csi_config_t as wifi_csi_acquire_config_t
 * and use a different wifi_csi_info_t layout. Fail the build loudly instead of
 * silently compiling a mismatched CSI configuration.
 */
#if CONFIG_SOC_WIFI_HE_SUPPORT
#error "This collector targets classic Wi-Fi SoCs (ESP32/S2/S3/C2/C3). On Wi-Fi 6 targets wifi_csi_config_t is wifi_csi_acquire_config_t with different fields: port the CSI configuration block in csi_receiver_enable() first."
#endif

/*
 * The noise floor bitfield of wifi_pkt_rx_ctrl_t only exists on targets that
 * report it; elsewhere it is an anonymous reserved bitfield.
 */
#if defined(CONFIG_IDF_TARGET_ESP32) || defined(CONFIG_IDF_TARGET_ESP32S3) || \
    defined(CONFIG_IDF_TARGET_ESP32C3) || defined(CONFIG_IDF_TARGET_ESP32C2)
#define CSI_TARGET_REPORTS_NOISE_FLOOR 1
#else
#define CSI_TARGET_REPORTS_NOISE_FLOOR 0
#endif

/** Wi-Fi/event-group bits used by the example application. */
#define CSI_WIFI_CONNECTED_BIT BIT0
#define CSI_WIFI_RETRY_BIT BIT1

/* ========================================================================== */
/* ==== BEGIN WIRE FORMAT (self-contained, host-testable) ==================== */
/* ========================================================================== */
/*
 * Everything between these markers depends only on <stdint.h>, <string.h> and the
 * frozen constants of csi_collector.h. It is deliberately free of ESP-IDF
 * dependencies so the exact byte layout can be compiled and verified on a host.
 */

/** One captured CSI frame travelling from the Wi-Fi task to the sender task. */
typedef struct {
    uint32_t sequence;      /*!< Sequence number assigned at capture time. */
    uint64_t timestamp_us;  /*!< Capture timestamp, microseconds since boot. */
    uint16_t len;           /*!< CSI payload length in bytes (2 * n_subcarriers). */
    uint8_t flags;          /*!< CSI_WIRE_FLAG_* bits. */
    uint8_t channel;        /*!< Primary channel the frame was received on. */
    int8_t rssi_dbm;        /*!< RSSI reported by the driver. */
    int8_t noise_floor_dbm; /*!< Noise floor reported by the driver (0 when unavailable). */
    int8_t buf[CSI_MAX_CSI_BYTES]; /*!< Verbatim copy of wifi_csi_info_t.buf. */
} csi_frame_msg_t;

/** Write a uint16 as two little-endian bytes. The wire buffer may be unaligned. */
static inline void csi_put_u16_le(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)(value & 0xFFu);
    p[1] = (uint8_t)((value >> 8) & 0xFFu);
}

/** Write a uint32 as four little-endian bytes. */
static inline void csi_put_u32_le(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)(value & 0xFFu);
    p[1] = (uint8_t)((value >> 8) & 0xFFu);
    p[2] = (uint8_t)((value >> 16) & 0xFFu);
    p[3] = (uint8_t)((value >> 24) & 0xFFu);
}

/** Write a uint64 as eight little-endian bytes. */
static inline void csi_put_u64_le(uint8_t *p, uint64_t value)
{
    for (unsigned i = 0; i < 8u; i++) {
        p[i] = (uint8_t)((value >> (8u * i)) & 0xFFu);
    }
}

/**
 * @brief Serialize one captured frame into the frozen wire format.
 *
 * Every field is written explicitly at its frozen offset: the collector never
 * copies a C struct onto the wire, because the compiler's padding, the target's
 * alignment rules and the host's endianness would all silently corrupt the
 * datagram.
 *
 * @param[out] out      Destination buffer of at least CSI_WIRE_HEADER_BYTES + msg->len bytes.
 * @param[in]  msg      Frame to serialize; msg->len must be even and <= CSI_MAX_CSI_BYTES.
 * @param[in]  node_id  Node identifier placed in header bytes 4..5.
 * @param[in]  sequence Sequence number placed in header bytes 8..11 (wraps at 2^32).
 *
 * @return The total number of bytes written (header + payload).
 */
static size_t csi_wire_serialize(uint8_t *out, const csi_frame_msg_t *msg, uint16_t node_id, uint32_t sequence)
{
    const uint16_t n_subcarriers = (uint16_t)(msg->len / 2u);

    csi_put_u16_le(&out[CSI_WIRE_OFF_MAGIC], CSI_WIRE_MAGIC);
    out[CSI_WIRE_OFF_VERSION] = (uint8_t)CSI_WIRE_VERSION;
    out[CSI_WIRE_OFF_FLAGS] = msg->flags;
    csi_put_u16_le(&out[CSI_WIRE_OFF_NODE_ID], node_id);
    csi_put_u16_le(&out[CSI_WIRE_OFF_PAYLOAD_BYTES], (uint16_t)msg->len);
    csi_put_u32_le(&out[CSI_WIRE_OFF_SEQUENCE], sequence);
    csi_put_u64_le(&out[CSI_WIRE_OFF_TIMESTAMP_US], msg->timestamp_us);
    out[CSI_WIRE_OFF_CHANNEL] = msg->channel;
    out[CSI_WIRE_OFF_RSSI_DBM] = (uint8_t)msg->rssi_dbm;
    out[CSI_WIRE_OFF_NOISE_FLOOR_DBM] = (uint8_t)msg->noise_floor_dbm;
    out[CSI_WIRE_OFF_N_SUBCARRIERS] = (uint8_t)n_subcarriers;

    /* Payload: verbatim driver buffer. buf[2*i] is imaginary, buf[2*i+1] is real. */
    if (msg->len > 0u) {
        memcpy(&out[CSI_WIRE_HEADER_BYTES], msg->buf, (size_t)msg->len);
    }

    return (size_t)CSI_WIRE_HEADER_BYTES + (size_t)msg->len;
}

/**
 * @brief IEEE 802.11 FCS (frame check sequence).
 *
 * CRC-32 with the 802.11 polynomial (0x04C11DB7, reflected 0xEDB88320), an
 * initial value of 0xFFFFFFFF and a final XOR of 0xFFFFFFFF - the same CRC that
 * Ethernet uses. The standard check value for the ASCII string "123456789" is
 * 0xCBF43926. On air the four FCS bytes are transmitted least-significant byte
 * first.
 *
 * @param[in] data Frame bytes (MAC header onwards, FCS excluded).
 * @param[in] len  Number of bytes.
 *
 * @return The FCS value for the frame.
 */
static uint32_t csi_fcs_crc32(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;

    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; bit++) {
            if (crc & 1u) {
                crc = (crc >> 1) ^ 0xEDB88320u;
            } else {
                crc >>= 1;
            }
        }
    }

    return ~crc;
}

/**
 * @brief Append the four FCS bytes (little-endian) of a frame to the frame buffer.
 *
 * @param[in,out] frame Frame buffer with room for four more bytes.
 * @param[in]     len   Current frame length in bytes.
 *
 * @return The new frame length (len + 4).
 */
static size_t csi_fcs_append(uint8_t *frame, size_t len)
{
    const uint32_t fcs = csi_fcs_crc32(frame, len);

    csi_put_u32_le(&frame[len], fcs);

    return len + 4u;
}

/* ========================================================================== */
/* ==== END WIRE FORMAT ===================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Collector state                                                            */
/* -------------------------------------------------------------------------- */

static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;

static volatile bool s_running;
static bool s_promiscuous_owned; /* true when the collector enabled promiscuous mode itself */
static volatile bool s_stop_requested;
static csi_collector_config_t s_config;
static QueueHandle_t s_queue;
static TaskHandle_t s_sender_task;
static TaskHandle_t s_illuminator_task;
static int s_sock = -1;
static uint32_t s_next_sequence;
static csi_collector_stats_t s_stats;

/* Counter helpers. The critical sections are a handful of instructions long and
 * are entered from ordinary task context: the CSI callback runs in the Wi-Fi
 * task, not in an ISR. */

static inline void csi_stats_inc_captured(void)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.frames_captured++;
    portEXIT_CRITICAL(&s_lock);
}

static inline void csi_stats_inc_dropped(void)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.frames_dropped++;
    portEXIT_CRITICAL(&s_lock);
}

static inline void csi_stats_inc_truncated(void)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.frames_truncated++;
    portEXIT_CRITICAL(&s_lock);
}

static inline void csi_stats_on_sent(uint32_t bytes)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.frames_sent++;
    s_stats.bytes_sent += (uint64_t)bytes;
    portEXIT_CRITICAL(&s_lock);
}

static inline void csi_stats_on_send_error(void)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.send_errors++;
    s_stats.frames_dropped++;
    portEXIT_CRITICAL(&s_lock);
}

static inline void csi_stats_on_sequence(uint32_t sequence)
{
    portENTER_CRITICAL(&s_lock);
    s_stats.sequence = sequence;
    portEXIT_CRITICAL(&s_lock);
}

csi_collector_stats_t csi_collector_get_stats(void)
{
    csi_collector_stats_t snapshot;

    portENTER_CRITICAL(&s_lock);
    snapshot = s_stats;
    portEXIT_CRITICAL(&s_lock);

    return snapshot;
}

static void csi_stats_reset(void)
{
    portENTER_CRITICAL(&s_lock);
    memset(&s_stats, 0, sizeof(s_stats));
    s_next_sequence = 0;
    portEXIT_CRITICAL(&s_lock);
}

bool csi_collector_is_running(void)
{
    return s_running;
}

/* -------------------------------------------------------------------------- */
/* Receiver path                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief CSI RX callback, executed in the Wi-Fi driver task context.
 *
 * This function must stay short and must never block: it copies the driver buffer
 * into a queue item and returns. No allocation, no logging, no socket I/O.
 */
static void csi_rx_callback(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;

    if (info == NULL || info->buf == NULL || info->len == 0u) {
        csi_stats_inc_dropped();
        return;
    }

    if (s_config.mac_filter_en && memcmp(info->mac, s_config.mac_filter, sizeof(s_config.mac_filter)) != 0) {
        return; /* Not the illuminator: filtered by configuration, not an error. */
    }

    csi_frame_msg_t msg;
    uint16_t len = info->len;
    uint8_t flags = 0u;

    if (len > CSI_MAX_CSI_BYTES) {
        /* Cannot happen on ESP32 silicon (HT40 tops out at 256 bytes), but the
         * datagram size is bounded unconditionally so that a future driver change
         * can never produce an oversized, fragmenting datagram. */
        len = CSI_MAX_CSI_BYTES;
        flags |= CSI_WIRE_FLAG_LAST_WORD_INVALID;
        csi_stats_inc_truncated();
    }
    if ((len & 1u) != 0u) {
        len--; /* CSI is a sequence of 2-byte (imag, real) pairs. */
        flags |= CSI_WIRE_FLAG_LAST_WORD_INVALID;
    }
    if (info->first_word_invalid) {
        flags |= CSI_WIRE_FLAG_FIRST_WORD_INVALID;
    }
    if (CSI_ASSUME_LAST_WORD_INVALID || (len % 4u) != 0u) {
        flags |= CSI_WIRE_FLAG_LAST_WORD_INVALID;
    }

    msg.timestamp_us = (uint64_t)esp_timer_get_time();
    msg.len = len;
    msg.flags = flags;
    msg.channel = (uint8_t)info->rx_ctrl.channel;
    msg.rssi_dbm = (int8_t)info->rx_ctrl.rssi;
#if CSI_TARGET_REPORTS_NOISE_FLOOR
    msg.noise_floor_dbm = (int8_t)info->rx_ctrl.noise_floor;
#else
    msg.noise_floor_dbm = 0; /* This target does not report a noise floor. */
#endif
    memcpy(msg.buf, info->buf, (size_t)len);

    portENTER_CRITICAL(&s_lock);
    msg.sequence = s_next_sequence++;
    portEXIT_CRITICAL(&s_lock);

    /* Zero block time: a full queue drops the frame instead of stalling the
     * Wi-Fi task. The sequence number has already been consumed, so the host
     * sees the loss as a gap in the sequence. */
    if (xQueueSend(s_queue, &msg, 0) != pdTRUE) {
        csi_stats_inc_dropped();
        return;
    }

    csi_stats_inc_captured();
}

/**
 * @brief Framing and upload task: pops captured frames and sends them over UDP.
 */
static void csi_sender_task(void *arg)
{
    (void)arg;

    csi_frame_msg_t msg;
    uint8_t datagram[CSI_MAX_DATAGRAM_BYTES];
    struct sockaddr_in dest = {0};
    uint32_t consecutive_errors = 0;

    dest.sin_family = AF_INET;
    dest.sin_port = htons(s_config.dest_port);
    dest.sin_addr.s_addr = s_config.dest_addr.addr;

    while (!s_stop_requested) {
        if (xQueueReceive(s_queue, &msg, pdMS_TO_TICKS(100)) != pdTRUE) {
            continue;
        }
        if (s_stop_requested) {
            break;
        }

        const int sock = s_sock;
        if (sock < 0) {
            csi_stats_inc_dropped();
            continue;
        }

        const size_t total = csi_wire_serialize(datagram, &msg, s_config.node_id, msg.sequence);
        const int sent = sendto(sock, datagram, total, 0, (struct sockaddr *)&dest, sizeof(dest));

        if (sent < 0) {
            csi_stats_on_send_error();
            /* Rate-limited: log the first failure and then every hundredth one. */
            consecutive_errors++;
            if (consecutive_errors == 1u || (consecutive_errors % 100u) == 0u) {
                ESP_LOGW(TAG, "sendto() failed (%lu consecutive failures, errno %d)",
                         (unsigned long)consecutive_errors, errno);
            }
            continue;
        }

        consecutive_errors = 0;
        csi_stats_on_sent((uint32_t)sent);
        csi_stats_on_sequence(msg.sequence);
    }

    s_sender_task = NULL;
    vTaskDelete(NULL);
}

/**
 * @brief Configure and enable CSI capture plus promiscuous mode.
 */
static esp_err_t csi_receiver_enable(void)
{
    /*
     * Classic Wi-Fi configuration. lltf/htltf/stbc_htltf2 select which training
     * fields contribute subcarriers; ltf_merge_en averages LLTF and HT-LTF on HT
     * packets; the channel filter is left off so adjacent subcarriers stay
     * independent, which is what the downstream DSP expects.
     */
    const wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    /* esp_wifi_set_csi_config() and esp_wifi_set_csi() both require Wi-Fi to be
     * started *or* promiscuous mode to be enabled - enable the sniffer first. */
    esp_err_t err = esp_wifi_set_promiscuous(true);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_promiscuous(true) failed: %s", esp_err_to_name(err));
        return err;
    }
    s_promiscuous_owned = true;

    err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_config() failed: %s", esp_err_to_name(err));
        goto fail;
    }

    err = esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_rx_cb() failed: %s", esp_err_to_name(err));
        goto fail;
    }

    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi(true) failed: %s", esp_err_to_name(err));
        goto fail_cb;
    }

    return ESP_OK;

fail_cb:
    esp_wifi_set_csi_rx_cb(NULL, NULL);
fail:
    if (s_promiscuous_owned) {
        esp_wifi_set_promiscuous(false);
        s_promiscuous_owned = false;
    }
    return err;
}

/**
 * @brief Disable CSI capture and promiscuous mode.
 */
static void csi_receiver_disable(void)
{
    (void)esp_wifi_set_csi(false);
    (void)esp_wifi_set_csi_rx_cb(NULL, NULL);
    if (s_promiscuous_owned) {
        (void)esp_wifi_set_promiscuous(false);
        s_promiscuous_owned = false;
    }
}

/* -------------------------------------------------------------------------- */
/* Transmitter path                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build the illuminator frame.
 *
 * Layout (48 bytes, no FCS - see CSI_TX_APPEND_FCS):
 *   0..23   minimal non-QoS 802.11 data frame:
 *             frame control 0x08 0x00 (version 0, type Data, subtype Data,
 *                                      ToDS = FromDS = 0, no other flags)
 *             duration 0x0000
 *             Addr1 = RA/DA = broadcast ff:ff:ff:ff:ff:ff
 *             Addr2 = TA = station interface MAC
 *             Addr3 = BSSID = station interface MAC
 *             sequence control, managed by this node
 *   24..31  LLC/SNAP header (aa aa 03 00 00 00 08 00)
 *   32..39  ASCII marker "SPCTFLOW" so receivers can recognise the illuminator
 *   40..43  little-endian frame counter
 *   44..47  reserved, zero
 *
 * @param[out] frame    Destination buffer of at least CSI_TX_FRAME_MAX_BYTES bytes.
 * @param[in]  src_mac  Transmitter interface MAC address (6 bytes).
 * @param[in]  counter  Frame counter carried inside the payload.
 * @param[in]  sequence 12-bit 802.11 sequence number.
 *
 * @return The frame length in bytes, FCS included only when CSI_TX_APPEND_FCS is set.
 */
static size_t csi_build_tx_frame(uint8_t *frame, const uint8_t *src_mac, uint32_t counter, uint16_t sequence)
{
    memset(frame, 0, CSI_TX_FRAME_BYTES);

    frame[0] = 0x08; /* Frame control: protocol 0, type Data (2), subtype Data (0). */
    frame[1] = 0x00; /* ToDS = 0, FromDS = 0, no retry/power/order bits. */
    /* frame[2..3]: duration, zero. */
    memset(&frame[4], 0xFF, 6);     /* Addr1: broadcast receiver address. */
    memcpy(&frame[10], src_mac, 6); /* Addr2: transmitter address. */
    memcpy(&frame[16], src_mac, 6); /* Addr3: BSSID (no distribution system). */

    /* Sequence control: 12-bit sequence number in the high bits, fragment 0. */
    frame[22] = (uint8_t)((sequence & 0x0Fu) << 4);
    frame[23] = (uint8_t)((sequence >> 4) & 0xFFu);

    /* LLC/SNAP header, so the frame is a well-formed Ethernet-tunnelling data frame. */
    static const uint8_t llc_snap[CSI_TX_LLC_SNAP_BYTES] = {0xAA, 0xAA, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00};
    memcpy(&frame[CSI_TX_MAC_HEADER_BYTES], llc_snap, sizeof(llc_snap));

    /* Recognisable marker plus a rolling counter. */
    static const char marker[CSI_TX_MARKER_BYTES] = {'S', 'P', 'C', 'T', 'F', 'L', 'O', 'W'};
    memcpy(&frame[CSI_TX_MAC_HEADER_BYTES + CSI_TX_LLC_SNAP_BYTES], marker, sizeof(marker));
    csi_put_u32_le(&frame[CSI_TX_MAC_HEADER_BYTES + CSI_TX_LLC_SNAP_BYTES + CSI_TX_MARKER_BYTES], counter);

    if (CSI_TX_APPEND_FCS) {
        return csi_fcs_append(frame, CSI_TX_FRAME_BYTES);
    }
    return CSI_TX_FRAME_BYTES;
}

/**
 * @brief Verify the FCS implementation against the standard CRC-32 check value.
 *
 * @return true when the routine produces 0xCBF43926 for "123456789".
 */
static bool csi_fcs_selftest(void)
{
    static const uint8_t check[] = "123456789";

    return csi_fcs_crc32(check, sizeof(check) - 1u) == 0xCBF43926u;
}

/**
 * @brief Illuminator task: transmit raw 802.11 frames at the configured rate.
 */
static void csi_illuminator_task(void *arg)
{
    (void)arg;

    uint8_t src_mac[6] = {0};
    esp_err_t err = esp_wifi_get_mac(WIFI_IF_STA, src_mac);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_get_mac() failed: %s", esp_err_to_name(err));
        s_illuminator_task = NULL;
        vTaskDelete(NULL);
        return;
    }

    uint8_t frame[CSI_TX_FRAME_MAX_BYTES];
    uint16_t sequence = 0;
    uint32_t counter = 0;
    const size_t frame_len = csi_build_tx_frame(frame, src_mac, counter, sequence);

    ESP_LOGI(TAG, "illuminator: %u-byte raw 802.11 data frame from " MACSTR ", rate %lu Hz",
             (unsigned)frame_len, MAC2STR(src_mac), (unsigned long)s_config.tx_frames_per_sec);
    ESP_LOGI(TAG, "illuminator: frame FCS is 0x%08lx (the MAC hardware computes and appends it on air)",
             (unsigned long)csi_fcs_crc32(frame, frame_len));

    TickType_t period = pdMS_TO_TICKS(1000u / s_config.tx_frames_per_sec);
    if (period == 0) {
        period = 1; /* Sub-tick rates fall back to one tick per frame. */
    }
    TickType_t last_wake = xTaskGetTickCount();
    uint32_t consecutive_errors = 0;

    while (!s_stop_requested) {
        counter++;
        sequence = (uint16_t)((sequence + 1u) & 0x0FFFu);
        csi_build_tx_frame(frame, src_mac, counter, sequence);

        err = esp_wifi_80211_tx(WIFI_IF_STA, frame, (int)frame_len, false);
        if (err != ESP_OK) {
            consecutive_errors++;
            portENTER_CRITICAL(&s_lock);
            s_stats.tx_errors++;
            portEXIT_CRITICAL(&s_lock);
            if (consecutive_errors == 1u || (consecutive_errors % 100u) == 0u) {
                ESP_LOGW(TAG, "esp_wifi_80211_tx() failed (%lu consecutive failures): %s",
                         (unsigned long)consecutive_errors, esp_err_to_name(err));
            }
        } else {
            consecutive_errors = 0;
            portENTER_CRITICAL(&s_lock);
            s_stats.frames_transmitted++;
            portEXIT_CRITICAL(&s_lock);
        }

        vTaskDelayUntil(&last_wake, period);
    }

    s_illuminator_task = NULL;
    vTaskDelete(NULL);
}

/* -------------------------------------------------------------------------- */
/* Channel handling                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Put the radio on the sensing channel.
 *
 * The transmitter never associates, so it simply forces the channel. A receiver
 * that is associated to an access point cannot choose its channel - the AP does -
 * so the actual channel is read back and a mismatch is reported to the user.
 */
static void csi_apply_channel(void)
{
    if (s_config.channel == 0u) {
        return; /* Channel 0 means "leave the radio where it already is". */
    }

    if (s_config.role == CSI_ROLE_RECEIVER) {
        wifi_ap_record_t ap_info = {0};
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            if (ap_info.primary != s_config.channel) {
                ESP_LOGW(TAG, "the associated AP is on channel %u but CSI_WIFI_CHANNEL is %u: the "
                              "illuminator must transmit on the channel the receiver is listening on",
                         (unsigned)ap_info.primary, (unsigned)s_config.channel);
            }
            return; /* Never move an associated station off its AP channel. */
        }
    }

    const wifi_second_chan_t second =
        (s_config.bandwidth == WIFI_BW_HT40) ? WIFI_SECOND_CHAN_ABOVE : WIFI_SECOND_CHAN_NONE;
    const esp_err_t err = esp_wifi_set_channel(s_config.channel, second);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_channel(%u) failed: %s", (unsigned)s_config.channel, esp_err_to_name(err));
    }
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

void csi_collector_config_defaults(csi_collector_config_t *config)
{
    if (config == NULL) {
        return;
    }

    memset(config, 0, sizeof(*config));
    config->node_id = 1;
    config->dest_addr.addr = ESP_IP4TOADDR(192, 168, 1, 100);
    config->dest_port = 5005;
    config->channel = 6;
    config->role = CSI_ROLE_RECEIVER;
    config->bandwidth = WIFI_BW_HT20;
    config->tx_frames_per_sec = 100;
    config->tx_phy_rate = WIFI_PHY_RATE_6M;
    config->mac_filter_en = false;
}

esp_err_t csi_collector_start(const csi_collector_config_t *config)
{
    if (config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_running) {
        ESP_LOGW(TAG, "collector already running");
        return ESP_ERR_INVALID_STATE;
    }
    if (config->role == CSI_ROLE_TRANSMITTER && config->tx_frames_per_sec == 0u) {
        ESP_LOGE(TAG, "tx_frames_per_sec must be >= 1");
        return ESP_ERR_INVALID_ARG;
    }
    if (config->role == CSI_ROLE_RECEIVER && config->dest_port == 0u) {
        ESP_LOGE(TAG, "dest_port must be non-zero");
        return ESP_ERR_INVALID_ARG;
    }
    if (config->channel < 1u || config->channel > 14u) {
        ESP_LOGW(TAG, "unusual Wi-Fi channel %u", (unsigned)config->channel);
    }

    s_config = *config;
    s_stop_requested = false;
    csi_stats_reset();

    if (s_config.role == CSI_ROLE_TRANSMITTER) {
        if (!csi_fcs_selftest()) {
            ESP_LOGE(TAG, "FCS self-test failed: the illuminator would emit malformed frames");
            return ESP_ERR_INVALID_STATE;
        }

        csi_apply_channel();

        const BaseType_t created = xTaskCreate(csi_illuminator_task, "csi_illum", CSI_ILLUMINATOR_TASK_STACK, NULL,
                                              CSI_ILLUMINATOR_TASK_PRIO, &s_illuminator_task);
        if (created != pdPASS) {
            s_illuminator_task = NULL;
            ESP_LOGE(TAG, "failed to create the illuminator task");
            return ESP_ERR_NO_MEM;
        }

        s_running = true;
        ESP_LOGI(TAG, "collector started in TRANSMITTER role: node_id %u, channel %u, %lu fps",
                 (unsigned)s_config.node_id, (unsigned)s_config.channel,
                 (unsigned long)s_config.tx_frames_per_sec);
        return ESP_OK;
    }

    /* Receiver role. */
    s_queue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(csi_frame_msg_t));
    if (s_queue == NULL) {
        ESP_LOGE(TAG, "failed to create the CSI queue (%u slots of %u bytes)", (unsigned)CSI_QUEUE_DEPTH,
                 (unsigned)sizeof(csi_frame_msg_t));
        return ESP_ERR_NO_MEM;
    }

    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "failed to create the UDP socket (errno %d)", errno);
        vQueueDelete(s_queue);
        s_queue = NULL;
        return ESP_FAIL;
    }

    /* Allow broadcast destinations; harmless (and ignored) for unicast ones. */
    int broadcast = 1;
    if (setsockopt(s_sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) != 0) {
        ESP_LOGD(TAG, "SO_BROADCAST not enabled (errno %d); unicast destinations still work", errno);
    }

    csi_apply_channel();

    if (csi_receiver_enable() != ESP_OK) {
        close(s_sock);
        s_sock = -1;
        vQueueDelete(s_queue);
        s_queue = NULL;
        return ESP_FAIL;
    }

    const BaseType_t created = xTaskCreate(csi_sender_task, "csi_udp", CSI_SENDER_TASK_STACK, NULL,
                                          CSI_SENDER_TASK_PRIO, &s_sender_task);
    if (created != pdPASS) {
        s_sender_task = NULL;
        csi_receiver_disable();
        close(s_sock);
        s_sock = -1;
        vQueueDelete(s_queue);
        s_queue = NULL;
        ESP_LOGE(TAG, "failed to create the UDP sender task");
        return ESP_ERR_NO_MEM;
    }

    s_running = true;

    ESP_LOGI(TAG, "collector started in RECEIVER role: node_id %u, channel %u, %s, target " IPSTR ":%u, queue depth %u",
             (unsigned)s_config.node_id, (unsigned)s_config.channel,
             s_config.bandwidth == WIFI_BW_HT40 ? "HT40" : "HT20", IP2STR(&s_config.dest_addr),
             (unsigned)s_config.dest_port, (unsigned)CSI_QUEUE_DEPTH);

    return ESP_OK;
}

esp_err_t csi_collector_stop(void)
{
    if (!s_running) {
        return ESP_OK;
    }

    s_stop_requested = true;

    if (s_config.role == CSI_ROLE_RECEIVER) {
        /* Stop producing before tearing the callback down, so that no frame is
         * queued into a queue that is about to be deleted. */
        csi_receiver_disable();
        vTaskDelay(pdMS_TO_TICKS(20)); /* Let an in-flight callback return. */
    }

    const TickType_t deadline = pdMS_TO_TICKS(CSI_TASK_STOP_TIMEOUT_MS);
    const TickType_t step = pdMS_TO_TICKS(10);
    TickType_t waited = 0;

    while ((s_sender_task != NULL || s_illuminator_task != NULL) && waited < deadline) {
        vTaskDelay(step);
        waited += step;
    }

    if (s_sender_task != NULL || s_illuminator_task != NULL) {
        ESP_LOGW(TAG, "a worker task did not stop within %d ms", (int)CSI_TASK_STOP_TIMEOUT_MS);
    }

    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
    if (s_queue != NULL) {
        vQueueDelete(s_queue);
        s_queue = NULL;
    }

    s_running = false;
    ESP_LOGI(TAG, "collector stopped");

    return ESP_OK;
}

/* ========================================================================== */
/* ==== APPLICATION ENTRY POINT ============================================= */
/* ========================================================================== */
/*
 * The code below is the default example application. Delete or disable it when
 * the collector is embedded into a bigger firmware image.
 */

static EventGroupHandle_t s_wifi_event_group;

static void csi_wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        /* Only the receiver associates. The illuminator must stay unassociated:
         * associating would drag it onto the AP's channel and make the driver
         * reject raw frames sent with en_sys_seq = false. */
        if (CSI_NODE_ROLE == CSI_ROLE_RECEIVER && CSI_WIFI_SSID[0] != '\0') {
            (void)esp_wifi_connect();
        }
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        const wifi_event_sta_disconnected_t *disc = (const wifi_event_sta_disconnected_t *)event_data;
        ESP_LOGW(TAG, "disconnected from \"%s\" (reason %d)", CSI_WIFI_SSID, disc != NULL ? disc->reason : -1);
        xEventGroupClearBits(s_wifi_event_group, CSI_WIFI_CONNECTED_BIT);
        /* Never block here: this handler runs in the system event task, which also
         * delivers the IP events. The reconnect itself is driven by app_main. */
        xEventGroupSetBits(s_wifi_event_group, CSI_WIFI_RETRY_BIT);
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        const wifi_event_sta_connected_t *conn = (const wifi_event_sta_connected_t *)event_data;
        if (conn != NULL) {
            ESP_LOGI(TAG, "associated to " MACSTR " on channel %u", MAC2STR(conn->bssid), (unsigned)conn->channel);
            if (conn->channel != CSI_WIFI_CHANNEL) {
                ESP_LOGW(TAG, "the AP is on channel %u while CSI_WIFI_CHANNEL is %d: put the AP or "
                              "CSI_WIFI_CHANNEL on the same channel as the illuminator",
                         (unsigned)conn->channel, CSI_WIFI_CHANNEL);
            }
        }
        return;
    }

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *got_ip = (const ip_event_got_ip_t *)event_data;
        if (got_ip != NULL) {
            ESP_LOGI(TAG, "got IPv4 address " IPSTR, IP2STR(&got_ip->ip_info.ip));
        }
        xEventGroupSetBits(s_wifi_event_group, CSI_WIFI_CONNECTED_BIT);
        return;
    }
}

/**
 * @brief Perform a pending reconnect attempt, if the station asked for one.
 *
 * Called from app_main (never from the event handler) so that the system event
 * task is never blocked.
 */
static void csi_app_service_wifi_retry(void)
{
    if (s_wifi_event_group == NULL || CSI_NODE_ROLE != CSI_ROLE_RECEIVER || CSI_WIFI_SSID[0] == '\0') {
        return;
    }
    if ((xEventGroupGetBits(s_wifi_event_group) & CSI_WIFI_RETRY_BIT) != 0) {
        xEventGroupClearBits(s_wifi_event_group, CSI_WIFI_RETRY_BIT);
        ESP_LOGI(TAG, "connecting to \"%s\"...", CSI_WIFI_SSID);
        (void)esp_wifi_connect();
    }
}

/**
 * @brief Bring up NVS, the network stack and Wi-Fi for the configured role.
 *
 * @return ESP_OK on success, or the first error encountered.
 */
static esp_err_t csi_app_init_network(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init() failed: %s", esp_err_to_name(err));
        return err;
    }

    ESP_ERROR_CHECK(esp_netif_init());

    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_event_loop_create_default() failed: %s", esp_err_to_name(err));
        return err;
    }

    s_wifi_event_group = xEventGroupCreate();
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "xEventGroupCreate() failed");
        return ESP_ERR_NO_MEM;
    }

    /* Both roles use the station interface: the illuminator never associates, but
     * it still needs the interface (and its netif glue) to emit raw frames. */
    if (esp_netif_create_default_wifi_sta() == NULL) {
        ESP_LOGE(TAG, "esp_netif_create_default_wifi_sta() failed");
        return ESP_FAIL;
    }

    wifi_init_config_t wifi_init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&wifi_init_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_init() failed: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_wifi_set_storage(WIFI_STORAGE_RAM);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_storage() failed: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, csi_wifi_event_handler, NULL, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "registering the WIFI_EVENT handler failed: %s", esp_err_to_name(err));
        return err;
    }
    err = esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, csi_wifi_event_handler, NULL, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "registering the IP_EVENT handler failed: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_mode() failed: %s", esp_err_to_name(err));
        return err;
    }

    if (CSI_NODE_ROLE == CSI_ROLE_RECEIVER) {
        wifi_config_t wifi_config = {0};
        memcpy(wifi_config.sta.ssid, CSI_WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1u);
        memcpy(wifi_config.sta.password, CSI_WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1u);
        /* WIFI_AUTH_OPEN as the threshold accepts any secured or open AP whose
         * credentials match, which is what a sensing node wants. */
        wifi_config.sta.threshold.authmode = WIFI_AUTH_OPEN;
        wifi_config.sta.pmf_cfg.capable = true;
        wifi_config.sta.pmf_cfg.required = false;
        err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_wifi_set_config() failed: %s", esp_err_to_name(err));
            return err;
        }
    } else {
        /*
         * Illuminator PHY rate. It must be configured after esp_wifi_init() and
         * before esp_wifi_start(). It is deliberately not touched in the receiver
         * role so that the receiver keeps the driver default for its uplink.
         */
        err = esp_wifi_config_80211_tx_rate(WIFI_IF_STA, CSI_TX_PHY_RATE);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "esp_wifi_config_80211_tx_rate() failed: %s (driver default is used)",
                     esp_err_to_name(err));
        }
    }

    err = esp_wifi_set_bandwidth(WIFI_IF_STA, CSI_WIFI_BANDWIDTH);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_bandwidth() failed: %s (driver default is used)", esp_err_to_name(err));
    }

    err = esp_wifi_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_start() failed: %s", esp_err_to_name(err));
        return err;
    }

    /* Power save must be off: modem sleep distorts CSI timing and timestamps. */
    err = esp_wifi_set_ps(WIFI_PS_NONE);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_ps(WIFI_PS_NONE) failed: %s", esp_err_to_name(err));
    }

    return ESP_OK;
}

/**
 * @brief Wait for an IPv4 address (receiver role with a configured SSID).
 *
 * @return ESP_OK once an address is assigned, ESP_ERR_TIMEOUT when the deadline
 *         expires, or ESP_OK immediately when the node is meant to stay
 *         unassociated.
 */
static esp_err_t csi_app_wait_for_ip(void)
{
    if (CSI_WIFI_SSID[0] == '\0') {
        ESP_LOGW(TAG, "CSI_WIFI_SSID is empty: the node stays unassociated on channel %d and cannot reach "
                      "the host. Set the SSID to stream CSI.",
                 CSI_WIFI_CHANNEL);
        return ESP_OK;
    }

    for (int attempt = 0; attempt < 30; attempt++) {
        const EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group, CSI_WIFI_CONNECTED_BIT, pdFALSE, pdFALSE,
                                                     pdMS_TO_TICKS(1000));
        if ((bits & CSI_WIFI_CONNECTED_BIT) != 0) {
            return ESP_OK;
        }
        csi_app_service_wifi_retry();
    }

    ESP_LOGE(TAG, "no IPv4 address after 30 s: check CSI_WIFI_SSID / CSI_WIFI_PASSWORD and that the AP is "
                  "reachable");
    return ESP_ERR_TIMEOUT;
}

static void csi_app_log_stats(void)
{
    const csi_collector_stats_t stats = csi_collector_get_stats();

    if (CSI_NODE_ROLE == CSI_ROLE_RECEIVER) {
        ESP_LOGI(TAG, "stats: captured=%lu sent=%lu dropped=%lu truncated=%lu bytes=%llu seq=%lu errors=%lu",
                 (unsigned long)stats.frames_captured, (unsigned long)stats.frames_sent,
                 (unsigned long)stats.frames_dropped, (unsigned long)stats.frames_truncated,
                 (unsigned long long)stats.bytes_sent, (unsigned long)stats.sequence,
                 (unsigned long)stats.send_errors);
    } else {
        ESP_LOGI(TAG, "stats: frames_tx=%lu tx_errors=%lu", (unsigned long)stats.frames_transmitted,
                 (unsigned long)stats.tx_errors);
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Spectraflow ESP32 CSI node - ESP-IDF %s, target %s", esp_get_idf_version(), CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, "role: %s, node_id: %d, channel: %d",
             CSI_NODE_ROLE == CSI_ROLE_TRANSMITTER ? "TRANSMITTER (illuminator)" : "RECEIVER", CSI_NODE_ID,
             CSI_WIFI_CHANNEL);

    if (csi_app_init_network() != ESP_OK) {
        ESP_LOGE(TAG, "network bring-up failed, the collector was not started");
        return;
    }

    csi_collector_config_t config;
    csi_collector_config_defaults(&config);
    config.node_id = CSI_NODE_ID;
    config.dest_port = CSI_DEST_PORT;
    config.channel = CSI_WIFI_CHANNEL;
    config.role = CSI_NODE_ROLE;
    config.bandwidth = CSI_WIFI_BANDWIDTH;
    config.tx_frames_per_sec = CSI_TX_FRAMES_PER_SEC;
    config.tx_phy_rate = CSI_TX_PHY_RATE;
    config.mac_filter_en = CSI_MAC_FILTER_EN;
    static const uint8_t mac_filter[6] = CSI_MAC_FILTER;
    memcpy(config.mac_filter, mac_filter, sizeof(config.mac_filter));

    if (config.role == CSI_ROLE_RECEIVER) {
        if (esp_netif_str_to_ip4(CSI_DEST_IP, &config.dest_addr) != ESP_OK) {
            ESP_LOGE(TAG, "CSI_DEST_IP \"%s\" is not a valid IPv4 address", CSI_DEST_IP);
            return;
        }
        if (csi_app_wait_for_ip() != ESP_OK) {
            ESP_LOGE(TAG, "cannot stream CSI without an IPv4 address, the collector was not started");
            return;
        }
    } else {
        ESP_LOGI(TAG, "illuminator runs unassociated on channel %d and needs no UDP uplink", CSI_WIFI_CHANNEL);
    }

    const esp_err_t err = csi_collector_start(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "csi_collector_start() failed: %s", esp_err_to_name(err));
        return;
    }

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(CSI_STATS_INTERVAL_MS));
        csi_app_service_wifi_retry();
        csi_app_log_stats();
    }
}
