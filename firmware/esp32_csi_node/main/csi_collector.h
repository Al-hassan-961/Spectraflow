/*
 * Spectraflow - ESP32 Wi-Fi Channel State Information (CSI) collector.
 *
 * Public API of the CSI collector component.
 *
 * The collector supports two roles, both compiled into a single binary and
 * selected at run time through csi_collector_config_t.role:
 *
 *   - CSI_ROLE_RECEIVER    : enables promiscuous-mode CSI capture, frames every
 *                            captured CSI buffer according to the frozen wire
 *                            format documented below and streams the resulting
 *                            datagrams over UDP to a host.
 *   - CSI_ROLE_TRANSMITTER : acts as a deterministic RF illuminator. It emits a
 *                            steady stream of minimal, unencrypted 802.11 data
 *                            frames with esp_wifi_80211_tx() so that a receiver
 *                            node always has something to measure. It does not
 *                            capture or upload anything.
 *
 * ---------------------------------------------------------------------------
 * FROZEN UDP WIRE FORMAT (do not change without bumping CSI_WIRE_VERSION)
 * ---------------------------------------------------------------------------
 * All multi-byte integers are LITTLE-ENDIAN. A datagram is a fixed 24-byte
 * header immediately followed by the CSI payload.
 *
 *   Offset  Size  Type    Field             Notes
 *   ------  ----  ------  ----------------  ---------------------------------
 *        0     2  uint16  magic             always 0x5F43 (wire bytes 43 5F)
 *        2     1  uint8   version           always 1
 *        3     1  uint8   flags             bit0 = first-word-invalid,
 *                                           bit1 = last-word-invalid
 *        4     2  uint16  node_id           configurable node identifier
 *        6     2  uint16  payload_bytes     == n_subcarriers * 2
 *        8     4  uint32  sequence          monotonic per datagram, wraps at 2^32
 *       12     8  uint64  timestamp_us      microseconds since boot
 *       20     1  uint8   channel           primary Wi-Fi channel
 *       21     1  int8    rssi_dbm          received signal strength
 *       22     1  int8    noise_floor_dbm   noise floor estimate
 *       23     1  uint8   n_subcarriers     number of subcarriers N, 1..128
 *       24   N*2   int8[]  csi payload       for each subcarrier i in [0, N-1]:
 *                                           buf[2*i]   = imaginary part
 *                                           buf[2*i+1] = real part
 *
 * The payload is a verbatim copy of the ESP32 wifi_csi_info_t buffer, which
 * uses exactly the imag/real interleaving above.
 *
 * An earlier draft of this table placed sequence (uint32) at offset 6 and
 * timestamp_us (uint64) at offset 8, which made the two fields overlap; that
 * draft is superseded by the layout above, which matches the native core
 * (core/include/csi_packet.hpp) and the Python parser
 * (spectraflow/ingestion/udp_receiver.py).
 *
 * The fields are laid out so that the wide members sit at well-aligned offsets,
 * which is why a naive "memcpy the struct" never worked here: the compiler's
 * padding (and the alignment rules) differ between the 32-bit Xtensa/RISC-V
 * target and any 64-bit host decoder, and a host would also have to fight its
 * own endianness. Every field is therefore written explicitly, byte by byte,
 * with the csi_put_*_le() helpers in csi_collector.c.
 *
 * Datagram size is bounded by CSI_MAX_DATAGRAM_BYTES (24 + 128*2 = 280 bytes),
 * far below the 1472-byte IPv4 MTU limit, so datagrams are never fragmented.
 * n_subcarriers is a single byte, which is sufficient because the cap is 128.
 *
 * Sequence semantics: the counter is consumed when a frame is captured, not when
 * it is transmitted, so a gap in `sequence` on the host means a frame really was
 * lost (full hand-off queue, or a sendto() failure).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_netif_ip_addr.h" /* esp_ip4_addr_t, ESP_IP4TOADDR */
#include "esp_wifi_types.h"    /* wifi_bandwidth_t, wifi_phy_rate_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Frozen wire format constants                                              */
/* ------------------------------------------------------------------------- */

/** Magic value in the first two bytes of every datagram (little-endian on the wire). */
#define CSI_WIRE_MAGIC 0x5F43u
/** Wire format version reported in header byte 2. */
#define CSI_WIRE_VERSION 1u
/** Size of the fixed header preceding the CSI payload. */
#define CSI_WIRE_HEADER_BYTES 24u

/** flags bit0: the first four bytes of the CSI buffer are invalid (hardware limitation). */
#define CSI_WIRE_FLAG_FIRST_WORD_INVALID 0x01u
/** flags bit1: the last four bytes of the CSI buffer are invalid. */
#define CSI_WIRE_FLAG_LAST_WORD_INVALID 0x02u

/* Header field offsets (bytes). */
#define CSI_WIRE_OFF_MAGIC 0u
#define CSI_WIRE_OFF_VERSION 2u
#define CSI_WIRE_OFF_FLAGS 3u
#define CSI_WIRE_OFF_NODE_ID 4u
#define CSI_WIRE_OFF_PAYLOAD_BYTES 6u
#define CSI_WIRE_OFF_SEQUENCE 8u
#define CSI_WIRE_OFF_TIMESTAMP_US 12u
#define CSI_WIRE_OFF_CHANNEL 20u
#define CSI_WIRE_OFF_RSSI_DBM 21u
#define CSI_WIRE_OFF_NOISE_FLOOR_DBM 22u
#define CSI_WIRE_OFF_N_SUBCARRIERS 23u

/**
 * @brief Maximum number of subcarriers carried in one datagram.
 *
 * 128 subcarriers == 256 payload bytes, which covers the largest CSI buffer the
 * ESP32 series delivers (HT40, 2 x 128 int8 values) and still fits the single
 * header byte that carries n_subcarriers. Anything larger is clipped, counted in
 * frames_truncated and flagged with CSI_WIRE_FLAG_LAST_WORD_INVALID rather than
 * being sent.
 */
#define CSI_MAX_SUBCARRIERS 128
/** Maximum CSI payload size in bytes (N * 2). */
#define CSI_MAX_CSI_BYTES (CSI_MAX_SUBCARRIERS * 2)
/** Maximum size of a complete datagram (header + payload). */
#define CSI_MAX_DATAGRAM_BYTES (CSI_WIRE_HEADER_BYTES + CSI_MAX_CSI_BYTES)

/* ------------------------------------------------------------------------- */
/* Public types                                                              */
/* ------------------------------------------------------------------------- */

/** @brief Role of this node. Both roles live in the same binary. */
typedef enum {
    CSI_ROLE_RECEIVER = 0,   /**< Capture CSI and stream it to the host over UDP. */
    CSI_ROLE_TRANSMITTER = 1 /**< Emit raw 802.11 frames as an RF illuminator. */
} csi_node_role_t;

/** @brief Collector configuration. Fill it with csi_collector_config_defaults() first. */
typedef struct {
    uint16_t node_id;         /**< Reported in the header; make it unique per node. */
    esp_ip4_addr_t dest_addr; /**< UDP destination address (host running Spectraflow). */
    uint16_t dest_port;       /**< UDP destination port. */
    uint8_t channel;          /**< Primary Wi-Fi channel, e.g. 1..13 for 2.4 GHz. 0 = leave as-is. */
    csi_node_role_t role;     /**< Receiver (capture) or transmitter (illuminator). */

    wifi_bandwidth_t bandwidth; /**< WIFI_BW_HT20 (64 subcarriers) or WIFI_BW_HT40 (up to 128). */

    uint32_t tx_frames_per_sec; /**< Transmitter role only: illuminator frame rate, default 100. */
    wifi_phy_rate_t tx_phy_rate;/**< Transmitter role only: PHY rate of the raw frames. */

    bool mac_filter_en;    /**< Receiver role only: capture CSI of one transmitter MAC only. */
    uint8_t mac_filter[6]; /**< Receiver role only: MAC accepted when mac_filter_en is true. */
} csi_collector_config_t;

/** @brief Runtime counters. All values are cumulative since the last start. */
typedef struct {
    uint32_t frames_captured;    /**< CSI buffers accepted from the Wi-Fi driver. */
    uint32_t frames_dropped;     /**< Frames lost: full queue, oversized/invalid buffer or socket error. */
    uint32_t frames_truncated;   /**< Frames whose CSI buffer exceeded CSI_MAX_CSI_BYTES and was clipped. */
    uint32_t frames_sent;        /**< Datagrams accepted by sendto(). */
    uint64_t bytes_sent;         /**< Total wire bytes accepted by sendto() (header + payload). */
    uint32_t sequence;           /**< Most recent sequence counter value sent on the wire (32-bit, wraps at 2^32). */
    uint32_t send_errors;        /**< sendto() failures. */
    uint32_t frames_transmitted; /**< Transmitter role: raw 802.11 frames handed to esp_wifi_80211_tx(). */
    uint32_t tx_errors;          /**< Transmitter role: esp_wifi_80211_tx() failures. */
} csi_collector_stats_t;

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

/**
 * @brief Fill a configuration structure with sane defaults.
 *
 * Defaults: node_id 1, destination 192.168.1.100:5005, channel 6, receiver role,
 * HT20, 100 frames/s at 6 Mbps for the transmitter role, MAC filter disabled.
 *
 * @param[out] config Configuration to initialize. Ignored when NULL.
 */
void csi_collector_config_defaults(csi_collector_config_t *config);

/**
 * @brief Start the collector.
 *
 * For the receiver role this enables promiscuous mode plus CSI capture, allocates
 * the lock-free-ish hand-off queue, creates the UDP socket and spawns the framing
 * task. For the transmitter role it starts the raw 802.11 frame illuminator.
 *
 * Wi-Fi itself must already be initialized and started by the caller, and for the
 * receiver role an IP address must already be configured on the default netif.
 *
 * @param[in] config Configuration, copied internally. Must not be NULL.
 *
 * @return
 *      - ESP_OK              on success
 *      - ESP_ERR_INVALID_ARG on a NULL or inconsistent configuration
 *      - ESP_ERR_INVALID_STATE if the collector is already running
 *      - ESP_ERR_NO_MEM      if the queue, task or socket could not be created
 *      - other esp_err_t values propagated from the Wi-Fi or lwIP APIs
 */
esp_err_t csi_collector_start(const csi_collector_config_t *config);

/**
 * @brief Stop the collector and release every resource it owns.
 *
 * Wi-Fi stays up; only CSI capture, promiscuous mode, the tasks, the queue and
 * the UDP socket are torn down. Safe to call when not running.
 *
 * @return
 *      - ESP_OK on success
 *      - other esp_err_t values from the Wi-Fi APIs used during teardown
 */
esp_err_t csi_collector_stop(void);

/**
 * @brief Read the current counters.
 *
 * Safe to call from any task at any time, including while the collector runs.
 *
 * @return A snapshot of the statistics.
 */
csi_collector_stats_t csi_collector_get_stats(void);

/**
 * @brief Query whether the collector is currently running.
 *
 * @return true when started and not yet stopped.
 */
bool csi_collector_is_running(void);

#ifdef __cplusplus
}
#endif
