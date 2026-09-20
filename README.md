# Spectraflow

Wi-Fi Channel State Information (CSI) sensing: **3D human pose visualisation,
contactless vital signs, and a WebGL front-end** — built from an ESP32 node, a
C++ DSP core and an async Python streaming server.

A pair of ESP32 boards illuminates a room with Wi-Fi frames; the receiver
captures per-subcarrier CSI; Spectraflow strips the phase corruption, cancels
the static multipath background, recovers respiration and heart rate from the
residual, estimates a 17-keypoint 3D skeleton, and streams all of it over a
WebSocket to a Three.js viewport.

> **Spectraflow reports what it can measure, and abstains when it cannot.**
> Every vital sign carries an SNR-derived confidence, and the estimator reports
> `null` (rendered as `--`) rather than a plausible-looking guess. An empty room
> produces no vitals at all — see [Limitations](#limitations) for the honest
> accuracy envelope.

---

## Architecture

```
  ESP32 TX node                                   ESP32 RX node
  (100 Hz illuminator)                            (promiscuous CSI capture)
        │                                                │
        └──────────── 802.11 frames ───────────────────►│
                                                         │ UDP, 24-byte header
                                                         ▼
  ┌──────────────────────────── spectraflow/ ────────────────────────────┐
  │ ingestion/  udp_receiver.py   parse datagram -> CsiFrame             │
  │             synthetic.py      physics-based frame generator          │
  │                                                                      │
  │ dsp/        phase_sanitizer   linear unwrap, CFO/SFO ramp removal    │
  │             clutter_removal   rolling background subtraction (M=100) │
  │             vitals_extractor  PCA -> Butterworth -> STFT -> SNR gate │
  │                                                                      │
  │ inference/  pose_estimator    ONNX Runtime, or CSI-driven analytic   │
  │                                                                      │
  │ server/     app.py            FastAPI or dependency-free ASGI + /ws  │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │ JSON over WebSocket
                                     ▼
  static/  index.html · scene.js (domes) · avatar.js (skeleton) · hud.js
```

The native C++ core (`core/`) accelerates the hot DSP primitives. It is an
**accelerator, never a requirement**: NumPy is the only hard runtime dependency,
and every routine has an equivalent pure-NumPy implementation selected
automatically at import time.

### Layout

| Path | Role |
|---|---|
| `firmware/esp32_csi_node/` | ESP-IDF v5 C firmware; transmitter and receiver roles in one binary |
| `core/include/` | `csi_packet.hpp` (wire format), `ring_buffer.hpp`, `dsp_filters.hpp` |
| `core/src/` | Packet codec, lock-free SPSC ring buffer, DSP implementations |
| `core/bindings.cpp` | pybind11 wrapper exposing the core as `spectraflow_core` |
| `spectraflow/ingestion/` | UDP transport, wire codec, synthetic CSI source |
| `spectraflow/dsp/` | Phase sanitization, clutter rejection, vital signs |
| `spectraflow/inference/` | CSI → 17-keypoint 3D pose |
| `spectraflow/server/` | ASGI application, WebSocket stream, static hosting |
| `static/` | Three.js viewport, skeleton renderer, glassmorphism HUD |
| `tests/` | Full suite; runs end-to-end with **no hardware** |

---

## Install & run

`requirements.txt` installs only what the pipeline truly needs — NumPy and
uvicorn — so it succeeds on every supported host. Everything else is an
accelerator that is detected at import time; skipping any of it changes nothing
but speed and which pose backend is used.

What each platform ends up with:

| | Linux | Termux (Android) |
|---|---|---|
| Native C++ core | ✅ builds | ✅ builds (auto-links `libpython`) |
| Server | FastAPI *or* dependency-free ASGI | dependency-free ASGI |
| Pose backend | ONNX Runtime **or** analytic | analytic |
| Verdict | full feature set | fully functional, software fallbacks |

---

### Linux

Works on any glibc/musl distro with Python ≥ 3.9. The native core needs a C++17
compiler and Python headers.

```bash
# Debian / Ubuntu
sudo apt install python3 python3-pip python3-dev build-essential cmake

# Fedora
#   sudo dnf install python3 python3-devel gcc-c++ cmake make
# Arch
#   sudo pacman -S python python-pip base-devel cmake

pip install -r requirements.txt

# Optional, and both have wheels here:
pip install pybind11 && pip install -e .   # native C++ core (faster DSP)
pip install onnxruntime                    # real neural pose inference
pip install fastapi                        # FastAPI flavour of the server
```

Run it — no hardware needed, this synthesises CSI:

```bash
python -m spectraflow.server --simulate
```

With an ESP32 receiver streaming to UDP:

```bash
python -m spectraflow.server --udp-port 5500 --port 8000
```

Then open <http://127.0.0.1:8000>.

> If the native core fails to build for any reason, the install still succeeds
> and prints a warning — the pure-NumPy path is used automatically.

---

### Termux (Android)

Everything runs on-device, no root, no proot. Install the toolchain first —
`clang` is required for the native core (`gcc` is not available).

```bash
pkg update && pkg upgrade
pkg install python clang cmake make git binutils termux-api

pip install -r requirements.txt
pip install pybind11 && pip install -e .   # native C++ core
```

(`termux-api` supplies `termux-open-url` and `termux-wake-lock` used below; the
server itself does not need it.)

Then run it and open the page **in the phone's own browser**:

```bash
python -m spectraflow.server --simulate
# in another Termux session, or just tap the link:
termux-open-url http://127.0.0.1:8000
```

Two Android-specific behaviours are handled automatically:

* **The native core is linked against `libpython`.** Termux's Python does not
  export its symbols to `dlopen`-ed extension modules, so a normally-linked
  build succeeds but fails at import with
  `cannot locate symbol "PyExc_ImportError"`. Both `setup.py` and
  `CMakeLists.txt` detect Android and link `libpython` explicitly. If you build
  the extension yourself, add `-lpython3.x`.
* **Do not try to install `onnxruntime` or `fastapi`.** Neither has a usable
  Android/aarch64 wheel (`onnxruntime` publishes none at all; FastAPI's
  `pydantic-core` dependency needs a Rust build that fails on Android). The
  server serves a dependency-free ASGI application with identical routes and
  payloads, and pose inference uses the analytic backend. `uvicorn` — a pure
  ASGI server, not a framework — is all that is required, and it installs fine.

> **Keeping it alive in the background:** Android aggressively reclaims
> Termux processes. Run the server under `termux-wake-lock` and disable battery
> optimisation for Termux if you want a long session.

---

### Options

Shared by both platforms.

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8000` | HTTP + WebSocket bind |
| `--udp-port` | `5500` | CSI datagram port |
| `--simulate` | off | synthesize CSI instead of binding UDP |
| `--window` | `5.0` | spectral analysis window, seconds |

`python -m spectraflow.server.app` works identically, as does the `spectraflow`
console script once the package is installed.

> **Opening it from another device?** The server binds `127.0.0.1` by default,
> so it is reachable only from the host itself. To browse from a phone or a
> laptop on the same network, bind all interfaces:
> `python -m spectraflow.server --simulate --host 0.0.0.0`, then browse to
> `http://<host-lan-ip>:8000`.

### Firmware (either platform)

Build with the Espressif toolchain, then flash one board as transmitter and one
as receiver:

```bash
idf.py set-target esp32 && idf.py build flash monitor
```

See `firmware/esp32_csi_node/README.md` for wiring, channel selection and the
destination IP/port configuration.

---

## The wire format

A frozen 24-byte little-endian header, shared byte-for-byte by the firmware, the
C++ core and the Python parser. All multi-byte integers are little-endian.

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 2 | uint16 | `magic` = `0x5F43` |
| 2 | 1 | uint8 | `version` = 1 |
| 3 | 1 | uint8 | `flags` (bit0 first-word-invalid, bit1 last-word-invalid) |
| 4 | 2 | uint16 | `node_id` |
| 6 | 2 | uint16 | `payload_bytes` = `n_subcarriers * 2` |
| 8 | 4 | uint32 | `sequence` |
| 12 | 8 | uint64 | `timestamp_us` |
| 20 | 1 | uint8 | `channel` |
| 21 | 1 | int8 | `rssi_dbm` |
| 22 | 1 | int8 | `noise_floor_dbm` |
| 23 | 1 | uint8 | `n_subcarriers` (1..128) |

Payload at offset 24: per subcarrier, `int8` **imag** then `int8` **real**.
Maximum datagram 280 bytes, well inside the 1472-byte MTU.

Every codec is byte-wise, never a struct cast: `timestamp_us` sits at offset 12,
which is 4-byte but not 8-byte aligned, so a plain (non-packed) struct would work
on the 32-bit firmware target and silently misparse in a 64-bit host build.

---

## WebSocket contract

`WS /ws` sends one JSON frame per message:

```json
{
  "type": "frame",
  "t": 1712345678.123,
  "seq": 12345,
  "node_id": 1,
  "presence": true,
  "motion": 0.12,
  "vitals": {
    "bpm": 72.4, "rpm": 15.2, "confidence": 0.74,
    "bpm_confidence": 0.6, "rpm_confidence": 0.84,
    "bpm_snr": 4.1, "rpm_snr": 22.4
  },
  "keypoints": [[x, y, z], "… 17 COCO-17 entries …"],
  "power": [-42.1, -41.0, "… one dB value per subcarrier …"]
}
```

Any field may be `null`. The client renders `--`; it never invents a value. The
server also sends `{"type":"hello"}` on connect. Coordinates are metres, Y-up,
origin at the floor centre of the sensing volume. `GET /api/status` returns
engine diagnostics; `GET /health` returns `{"status":"ok"}`.

---

## How the DSP works

1. **Phase sanitization.** Commodity receivers do not preserve absolute phase.
   CFO adds a constant rotation; SFO adds a term growing *linearly with
   subcarrier index*. Both are removed by detrending the unwrapped phase:
   `phase[i] - ((phase[N-1] - phase[0]) / (N-1)) * i - mean(phase)`. This is what
   makes phase-based sensing possible without a calibrated receiver.

2. **Clutter rejection.** The static environment dominates the raw CSI by one to
   two orders of magnitude. `ClutterRemover` cancels it by subtracting a trailing
   complex mean over M = 100 frames (5 s at 20 Hz).

   It is deliberately **not** wired in front of the vital-sign estimator, and
   that is a measured decision rather than an oversight. Phase error scales as
   `1/|H|`, and clutter removal shrinks `|H|` by ~19× while leaving the absolute
   noise floor untouched — so `arg(H)` comes out **56× noisier** than it went in,
   even though the static reflector is gone. Feeding cleaned CSI into the
   vitals pipeline drops subject respiration SNR from ~18 dB to ~1 dB and makes
   the rate undetectable. The estimator instead removes clutter by detrending
   *within* its analysis window, which cancels the background without shrinking
   the magnitude used to form the phase. `tests/test_dsp.py` pins this.

3. **Sensing signal.** Subject motion is *common mode* across subcarriers;
   receiver noise is independent per subcarrier. The window is band-limited to
   0.1–2.5 Hz and projected onto its **first principal component**, which is the
   maximum-likelihood common-mode waveform. Band-limiting before the projection
   matters — environmental drift is also common mode and would otherwise
   dominate the first component.

4. **Vitals.** A 2nd-order Butterworth band-pass (bilinear transform with
   pre-warping) isolates each band — 0.1–0.5 Hz (6–30 RPM) and 0.8–2.5 Hz
   (48–150 BPM) — before a Hann-windowed FFT peaks it. The peak is refined by
   parabolic interpolation to sub-bin accuracy, without which a 5 s window
   quantises the estimate to 12 breaths/min.

Three details are what separate a working estimator from a plausible-looking
one, and each was added in response to an observed failure:

* **The noise floor is broadband, not in-band.** Scoring a peak against the
  spectrum it was filtered from is circular — the filter already removed those
  bins, so the ratio is inflated for *any* input. Measured against a broadband
  floor, a real subject sits at 37–42 dB and an empty room at 10–23 dB.
* **Respiration harmonics are notched out of the heart band.** Respiration is
  10–20× stronger than the cardiac signal, so a 16 RPM breath puts its 3rd
  harmonic at 0.81 Hz — squarely inside the heart band, where it was being
  reported as a 48 BPM heart rate.
* **Rates are gated on cross-window stability.** Slow environmental drift also
  produces a strong low-frequency peak that no single 5 s window can distinguish
  from breathing. A physiological rhythm holds its frequency; a drift peak
  wanders. The reported value is the median of recent windows and is withheld
  until they agree within 15 %.

---

## Verification

```bash
pytest tests/ -q
```

**111 tests pass** (1 skipped without `onnxruntime`), with no hardware attached.
The suite covers the wire codec byte-for-byte, malformed-input rejection, the
real UDP transport over loopback, DSP correctness, backend equivalence, the
WebSocket contract, path-traversal refusal, and client backpressure.

It also asserts the negative results that matter most:

| Scenario | Assertion |
|---|---|
| Empty room, 8 seeds | Reports **no** BPM and **no** RPM |
| Realistic subject | Respiration recovered within 2.5 RPM |
| Realistic subject | **Never** a confidently wrong heart rate |
| High-SNR subject | Heart rate recovered within 4 BPM |
| Static channel | Pose confidence ≈ 0 |

Both backends are exercised: run the suite with the native module hidden and it
reports `102 passed, 10 skipped`, with numeric results matching.

---

## Limitations

Stated plainly, because a sensing system that overstates itself is worse than one
that abstains.

* **Heart rate is not reliably recoverable at realistic amplitudes.** Cardiac
  chest excursion (~0.6 mm) is roughly 8× smaller than respiration (~5 mm), and
  even after principal-component extraction the heart band is dominated by
  respiration harmonics and noise. At realistic amplitudes Spectraflow usually
  reports `bpm: null` — which is the correct answer, not a defect. Detection
  becomes reliable at high SNR (verified with a 12 mm cardiac excursion and a
  20 s window). Treat BPM as a best-effort indicator.
* **A 5 s window is short for these bands.** Frequency resolution is 0.2 Hz
  (12 breaths/min); sub-bin interpolation recovers the peak location well but
  cannot manufacture information. `--window 20` markedly improves both accuracy
  and confidence.
* **Pose estimation without a model is a motion visualisation, not anatomy.**
  No `onnxruntime` wheel exists for Android, so the default path is a
  deterministic estimator that projects CSI features onto a latent space and
  perturbs a canonical skeleton. It genuinely responds to the channel, but it
  has no learned model of the human body. Supply a real model with
  `SpectraflowConfig.onnx_model_path` for metrically meaningful joints; the
  wired contract is input `(1, 1, subcarriers, window)`, output `(17, 3)`.
* **One transmitter/receiver pair senses one room.** Multi-node fusion is not
  implemented.
* **Not a medical device.** Reported rates are uncalibrated estimates.

---

## Author

**Al-hassan Shehade** — Spectraflow. MIT licensed.
