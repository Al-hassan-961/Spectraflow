# Spectraflow ESP32 CSI node

ESP-IDF v5.x firmware component that captures Wi-Fi **Channel State Information (CSI)**
on an ESP32 and streams it to a host over UDP using a frozen, byte-exact wire format.

One binary supports two roles, selected in the configuration block at the top of
`main/csi_collector.c`:

| Role | `CSI_NODE_ROLE` | What it does |
|------|-----------------|--------------|
| **Receiver** | `CSI_ROLE_RECEIVER` | Enables promiscuous mode + CSI capture, frames every CSI buffer into a 24-byte header + payload datagram and sends it to the host over UDP. |
| **Transmitter** | `CSI_ROLE_TRANSMITTER` | Deterministic RF illuminator: emits minimal, unencrypted 802.11 data frames with `esp_wifi_80211_tx()` (default 100 Hz). It never associates and never uploads. |

```
firmware/esp32_csi_node/
├── CMakeLists.txt          ESP-IDF project wrapper
├── sdkconfig.defaults      CSI + CPU + log level + driver buffer defaults
├── README.md               this file
└── main/
    ├── CMakeLists.txt      idf_component_register()
    ├── csi_collector.h     public API + frozen wire format documentation
    └── csi_collector.c     collector implementation + example app_main()
```

---

## 1. Requirements

* ESP-IDF **v5.x** (`idf.py --version`). Tested API surface against v5.0/v5.1/v5.5 headers.
* A classic Wi-Fi target: **ESP32**, ESP32-S2/S3/C2/C3.
  Wi-Fi 6 (HE) targets such as ESP32-C6/C5 are **rejected at compile time** with an
  explicit `#error`: on those SoCs `wifi_csi_config_t` is
  `wifi_csi_acquire_config_t` with a different field set and `wifi_csi_info_t`
  differs as well, so the CSI configuration block must be ported first.
* Two boards for a complete link (one transmitter, one receiver) and a host on the
  same LAN as the receiver.

## 2. Build and flash

```bash
cd firmware/esp32_csi_node

idf.py set-target esp32          # esp32s3 / esp32c3 also work
idf.py build                     # apply sdkconfig.defaults on the first build
idf.py -p /dev/ttyUSB0 flash monitor
```

`idf.py set-target` must be run once before the first build. If you ever change the
target, run it again (it resets `sdkconfig`).

`idf.py menuconfig` is optional - the whole configuration lives in the `#define`
block at the top of `main/csi_collector.c`. Check
`Component config -> Wi-Fi -> WiFi CSI(Channel State Information)` if you want to
confirm that CSI support is on.

## 3. Configure

Edit the **USER CONFIGURATION** block at the top of `main/csi_collector.c`:

```c
#define CSI_NODE_ROLE CSI_ROLE_RECEIVER   // or CSI_ROLE_TRANSMITTER
#define CSI_NODE_ID 1                     // unique per node
#define CSI_DEST_IP "192.168.1.100"       // host running Spectraflow (receiver only)
#define CSI_DEST_PORT 5005                // UDP port (receiver only)
#define CSI_WIFI_CHANNEL 6                // both nodes must use this channel
#define CSI_WIFI_BANDWIDTH WIFI_BW_HT20   // HT20 -> up to 64 subcarriers
#define CSI_WIFI_SSID "YOUR_AP_SSID"      // receiver only
#define CSI_WIFI_PASSWORD "YOUR_AP_PASSWORD"
#define CSI_TX_FRAMES_PER_SEC 100         // transmitter only
#define CSI_TX_PHY_RATE WIFI_PHY_RATE_6M
#define CSI_MAC_FILTER_EN 0               // 1 = capture one transmitter MAC only
#define CSI_MAC_FILTER {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01}
```

| Setting | Applies to | Notes |
|---|---|---|
| `CSI_DEST_IP` / `CSI_DEST_PORT` | receiver | Destination IPv4 address parsed with `esp_netif_str_to_ip4()`. Broadcast addresses work (`SO_BROADCAST` is enabled on the socket). Ignored by the transmitter. |
| `CSI_WIFI_CHANNEL` | both | The illuminator, the receiver and the AP the receiver associates with must all sit on this channel. A mismatch is logged as a warning. |
| `CSI_WIFI_SSID` / `_PASSWORD` | receiver | Leave empty to run unassociated on `CSI_WIFI_CHANNEL`: CSI is still captured, but nothing can be uploaded because the node has no IP address. |
| `CSI_TX_FRAMES_PER_SEC` | transmitter | 100 by default; `sdkconfig.defaults` sets `CONFIG_FREERTOS_HZ=1000` so the period is a clean 10 ms. |
| `CSI_MAC_FILTER_EN` | receiver | Set to 1 and put the illuminator MAC (printed on the transmitter's console at boot) into `CSI_MAC_FILTER` to ignore ambient traffic. |
| `CSI_WIFI_BANDWIDTH` | both | HT20 gives up to 64 subcarriers; HT40 up to 128. The illuminator sends legacy OFDM frames, so HT20 is the practical default. |

Recommended: give each node its own `CSI_NODE_ID` (`1`, `2`, ...) so multi-node captures
can be separated on the host.

## 4. Flash a transmitter node and a receiver node

The same project is flashed twice with a different `CSI_NODE_ROLE`.

**Transmitter node**

```c
#define CSI_NODE_ROLE CSI_ROLE_TRANSMITTER
#define CSI_NODE_ID 2
#define CSI_WIFI_CHANNEL 6
#define CSI_TX_FRAMES_PER_SEC 100
```

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

Expected boot output:

```
I (xxx) csi_collector: Spectraflow ESP32 CSI node - ESP-IDF v5.1.7-dirty, target esp32
I (xxx) csi_collector: role: TRANSMITTER (illuminator), node_id: 2, channel: 6
I (xxx) csi_collector: illuminator runs unassociated on channel 6 and needs no UDP uplink
I (xxx) csi_collector: illuminator: 48-byte raw 802.11 data frame from 1a:5f:43:00:00:02, rate 100 Hz
I (xxx) csi_collector: illuminator: frame FCS is 0x054E4750 (the MAC hardware computes and appends it on air)
I (xxx) csi_collector: collector started in TRANSMITTER role: node_id 2, channel 6, 100 fps
I (xxx) csi_collector: stats: frames_tx=500 tx_errors=0
```

Note the MAC address: it is the value to put into the receiver's `CSI_MAC_FILTER`.

**Receiver node**

```c
#define CSI_NODE_ROLE CSI_ROLE_RECEIVER
#define CSI_NODE_ID 1
#define CSI_DEST_IP "192.168.1.100"   // your host
#define CSI_DEST_PORT 5005
#define CSI_WIFI_CHANNEL 6            // same channel as the transmitter
#define CSI_WIFI_SSID "YOUR_AP_SSID"
#define CSI_WIFI_PASSWORD "YOUR_AP_PASSWORD"
```

```bash
idf.py -p /dev/ttyUSB1 flash monitor
```

Expected boot output:

```
I (xxx) csi_collector: associated to 24:xx:xx:xx:xx:xx on channel 6
I (xxx) csi_collector: got IPv4 address 192.168.1.37
I (xxx) csi_collector: collector started in RECEIVER role: node_id 1, channel 6, HT20, target 192.168.1.100:5005, queue depth 16
I (xxx) csi_collector: stats: captured=500 sent=500 dropped=0 truncated=0 bytes=140000 seq=499 errors=0
```

`captured` counts CSI buffers taken from the driver, `sent` counts datagrams accepted by
`sendto()`, `dropped` counts frames lost to a full queue or a socket error, and `seq` is
the most recent sequence number written to the wire. A steadily rising `dropped` means the
host or the Wi-Fi link cannot keep up, not that CSI capture failed.

**Verify on the host**

```bash
# crudest possible check: count datagrams and show the first one
python3 - <<'EOF'
import socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5005))
while True:
    data, addr = s.recvfrom(2048)
    magic, ver, flags, node, pbytes = struct.unpack_from("<HBBHH", data, 0)   # 0,2,3,4,6
    seq, = struct.unpack_from("<I", data, 8)                                  # 8
    ts, = struct.unpack_from("<Q", data, 12)                                  # 12
    ch, rssi, nf, nsc = struct.unpack_from("<BbbB", data, 20)                 # 20,21,22,23
    assert len(data) == 24 + pbytes == 24 + nsc * 2
    print(f"{addr[0]} node={node} seq={seq} magic=0x{magic:04X} v{ver} flags=0x{flags:02X} "
          f"ts={ts}us ch={ch} rssi={rssi} noise={nf} N={nsc} payload={pbytes} total={len(data)}")
EOF
```

## 5. Typical wiring (2 boards: one TX, one RX)

```
        channel 6                         channel 6
   ┌───────────────┐   raw 802.11    ┌───────────────┐   UDP :5005   ┌──────────────┐
   │  ESP32 #2     │  data frames    │  ESP32 #1     │  CSI stream   │  Host        │
   │  TRANSMITTER  │ ───────────────▶│  RECEIVER     │ ─────────────▶│  Spectraflow │
   │  (illuminator)│    100 Hz       │  STA + CSI    │   100 Hz      │  192.168.1.100│
   └───────────────┘                 └───────┬───────┘               └──────┬───────┘
         no Wi-Fi association                 │                              │
                                       associates to the AP  ◀──────────────┘
                                       (same channel 6)        same LAN
```

* The **transmitter** never associates. It just sits on `CSI_WIFI_CHANNEL` and blasts
  frames, which removes every association, encryption and sequence-control side effect
  from the raw-TX path.
* The **receiver** associates to your normal AP so that it has an IP route to the host,
  and *that AP must operate on the same channel as the illuminator*. Any 2.4 GHz AP on a
  fixed channel works; set `CSI_WIFI_CHANNEL` to the AP's channel and, ideally, pin the AP
  to that channel in its own settings.
* The host must be on the same LAN as the receiver (or be reachable from it) and must be
  listening on `CSI_DEST_PORT` (UDP).
* Optional: the receiver also captures whatever ambient traffic is on the channel. With
  `CSI_MAC_FILTER_EN 1` and the illuminator MAC, only illuminator CSI is streamed.

## 6. UDP wire format (frozen)

All multi-byte integers are **little-endian**. A datagram is a fixed **24-byte header**
immediately followed by the CSI payload. The layout places the wide members at
well-aligned offsets, but the datagram is still assembled **byte by byte** with explicit
little-endian helpers - a C struct is never copied onto the wire, because the compiler's
padding, the target's alignment rules and a 64-bit host's own endianness would all
silently corrupt the stream.

| Offset | Size | Type   | Field             | Notes |
|--------|------|--------|-------------------|-------|
| 0      | 2    | uint16 | magic             | always `0x5F43` (`43 5F` on the wire) |
| 2      | 1    | uint8  | version           | always `1` |
| 3      | 1    | uint8  | flags             | bit0 = first-word-invalid, bit1 = last-word-invalid |
| 4      | 2    | uint16 | node_id           | configurable node identifier |
| 6      | 2    | uint16 | payload_bytes     | `== n_subcarriers * 2` |
| 8      | 4    | uint32 | sequence          | monotonic per datagram, wraps at 2^32 |
| 12     | 8    | uint64 | timestamp_us      | microseconds since boot (`esp_timer_get_time()` at capture) |
| 20     | 1    | uint8  | channel           | primary Wi-Fi channel |
| 21     | 1    | int8   | rssi_dbm          | RSSI reported by the driver |
| 22     | 1    | int8   | noise_floor_dbm   | noise floor estimate (0 on targets that do not report it) |
| 23     | 1    | uint8  | n_subcarriers     | number of subcarriers `N`, 1..128 |
| 24     | N*2  | int8[] | csi payload       | for each subcarrier `i`: `buf[2*i]` = imaginary, `buf[2*i+1]` = real |

* The payload is a verbatim copy of the ESP32 `wifi_csi_info_t.buf` buffer, which already
  uses the imag/real interleaving above.
* Maximum datagram size is `CSI_MAX_DATAGRAM_BYTES` = 24 + 128*2 = **280 bytes**, far below
  the 1472-byte IPv4 MTU limit, so datagrams are never fragmented. Anything larger than
  `CSI_MAX_SUBCARRIERS` (128) is clipped, counted in `frames_truncated` and flagged with
  bit1 of `flags`. `n_subcarriers` is a single byte, which is enough because the cap is 128.
* `sequence` gaps on the host mean real loss: the counter is consumed at capture time, so a
  frame dropped because the hand-off queue was full still burns a sequence number.
* `sequence` is a full 32-bit counter on the wire (`payload_bytes` sits between `node_id`
  and `sequence` so that the wide fields stay well aligned).

> An earlier draft of this table placed `sequence` (uint32) at offset 6 and `timestamp_us`
> (uint64) at offset 8, where the two fields overlap. That draft is **superseded** and is
> not implemented; the offsets above are the ones this firmware emits and the ones a host
> decoder must use.

## 7. Behaviour and performance notes

* **CSI callback.** Runs in the Wi-Fi driver task. It copies the driver buffer into a
  pre-allocated queue item and returns: no allocation, no logging, no socket I/O, no
  blocking. The queue is fed with `xQueueSend(..., 0)` - the `FromISR` variants are wrong
  here because the callback is a task, not an ISR.
* **Drop instead of block.** A full queue (default 16 slots) drops the frame, increments
  `frames_dropped` and leaves a gap in the sequence. The Wi-Fi task is never stalled.
* **Framing task.** A lower-priority task pops frames, serializes them into a reusable
  stack buffer and calls `sendto()` on a persistent UDP socket created once at start.
  Send errors are counted and logged at most once per 100 consecutive failures.
* **Payload memory.** The queue costs about `CSI_QUEUE_DEPTH * ~272 bytes` (~4.4 KB).
  `CONFIG_ESP_WIFI_CSI_ENABLED` additionally costs roughly
  `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM` KB of RAM.
* **Power save** is forced off (`WIFI_PS_NONE`): modem sleep distorts CSI timing and the
  `rx_ctrl.timestamp` field. `sdkconfig.defaults` keeps light sleep and tickless idle off.
* **Sniffer + association.** ESP-IDF allows promiscuous mode while associated, but the
  sniffer has a "great impact" on the throughput of that connection. The CSI uplink here is
  only ~28 kB/s at 100 Hz, so this is not a problem in practice - just do not run bulk
  traffic over the receiver's uplink at the same time.
* **First/last word validity.** `wifi_csi_info_t.first_word_invalid` sets bit0 of `flags`.
  The driver exposes no matching flag for the tail, so bit1 is raised when the delivered
  length is not a multiple of 4 bytes (an incomplete final 32-bit word) or when
  `CSI_ASSUME_LAST_WORD_INVALID` is defined as 1 at build time.

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `captured=0` on the receiver | Receiver and illuminator are on different channels, or the receiver's AP moved to another channel. Both are logged at boot. |
| `dropped` climbing steadily | Host or uplink too slow, or the queue is too small for the traffic burst - raise `CSI_QUEUE_DEPTH`, or filter ambient traffic with `CSI_MAC_FILTER_EN`. |
| `sendto() failed (...)` warnings | No IP address / no route to the host: check `CSI_DEST_IP`, that the receiver got an IPv4 address, and that the host's UDP port is open. |
| `no IPv4 address after 30 s` | Wrong `CSI_WIFI_SSID` / `CSI_WIFI_PASSWORD`, AP out of range, or a hidden/enterprise AP. |
| Illuminator `frames_tx=0`, `tx_errors` rising | Wi-Fi is not started, or the interface/target combination rejected the raw frame (see the notes below). |
| Compile error about `wifi_csi_acquire_config_t` | You are building for a Wi-Fi 6 target; port the CSI configuration block first (see Requirements). |
| `esp_wifi_80211_tx() failed ... ESP_ERR_INVALID_ARG` | The driver rejects raw frames once a connection exists with `en_sys_seq=false`. The illuminator never associates, so this points at modified Wi-Fi bring-up code. |

## 9. Notes for a reviewer / integrator

* `esp_wifi_80211_tx()` is fed a **48-byte frame without the FCS**: the API documents a
  minimum length of 24 bytes (a bare MAC header) and the MAC hardware computes and appends
  the FCS on air. `csi_fcs_crc32()` in `csi_collector.c` is a complete 802.11 FCS (CRC-32,
  reflected, init/final `0xFFFFFFFF`) whose check value is verified at startup against the
  standard `0xCBF43926`; the illuminator logs the FCS it would append. Set
  `CSI_TX_APPEND_FCS 1` at build time only if a driver revision is verified to require the
  FCS inside the buffer.
* The illuminator runs in **station mode without associating**, which is the
  "no Wi-Fi connection" scenario of the ESP-IDF raw-frame guide: no ToDS/FromDS
  restrictions, no encryption, and `en_sys_seq = false` with a self-managed 12-bit
  sequence number are all valid there.
* `esp_wifi_set_csi_config()` / `esp_wifi_set_csi()` require Wi-Fi to be started **or**
  promiscuous mode to be enabled; the collector enables promiscuous mode first, then
  configures CSI, then registers the callback, then enables capture - and unwinds in the
  opposite order on failure.
* `esp_wifi_config_80211_tx_rate()` must be called **after `esp_wifi_init()` and before
  `esp_wifi_start()`**; the transmitter role does exactly that, the receiver leaves the
  setting alone.
* `esp_wifi_set_channel()` requires Wi-Fi to be started, and an associated station must
  never be moved off its AP's channel - the collector only forces the channel when the
  station is unassociated and otherwise logs a channel mismatch.
* `csi_collector.c` also contains the example `app_main()`. Everything above the
  `APPLICATION ENTRY POINT` banner is the reusable component; delete the part below it to
  embed the collector in a larger firmware image.
