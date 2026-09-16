# AI Glass Integration — Architecture and File Map

How BrailleLens drives the AI Glass hardware, why the transport looks the way it
does, and what each new or changed file is responsible for.

Read the first section before touching any of this code. The transport is not
what most Flutter Bluetooth work assumes, and the rest of the design only makes
sense once that is clear.

---

## 1. The pivot: this is not BLE GATT

The obvious way to add smart-glasses support to a Flutter app is
`flutter_blue_plus`: scan for the peripheral, discover services, subscribe to
the characteristic that reports button presses. **That approach cannot work
here, and it is worth understanding why before you try it.**

The AI Glass is a **Realtek Audio Connect** device. It exposes three separate
transports, none of which is GATT:

| Concern | Transport | What carries it |
| --- | --- | --- |
| Control, button, mic PCM, stills, battery | **Bluetooth Classic** — vendor protocol over SPP | `bbpro` `TransportLayer` → `SmartWearModelClient` |
| Live video | **Wi-Fi RTSP** at `rtsp://<ip>:554` | `RTKLiveStreamingManager`, then an RTSP pull (we play it with `media_kit`/libmpv) |
| Voice prompts / TTS, mic audio | **Bluetooth A2DP + HFP/SCO** | Standard Android `AudioManager` |

Consequences that shape every file below:

- **There is no button characteristic.** The frame button arrives as
  `SmartWearModelCallback.onDeviceTriggeredTakePhoto()` on the vendor channel.
  Nothing in the BLE stack will ever see it.
- **The SDK is a native Android AAR.** It has no Dart binding and no Maven
  coordinates, so it is vendored into `android/app/libs/` and reached through
  platform channels. The Dart layer is a *face* over that bridge, not a
  protocol implementation.
- **Video does not come over Bluetooth.** Asking for a camera preview means
  bringing up a Wi-Fi link and pulling RTSP — a different subsystem, with its
  own failure modes, from the one that delivers the button press.

Every SDK constant and callback signature used in this integration was verified
with `javap` against `rtk-audioconnect-smartwear-1.8.41.aar` rather than copied
from documentation.

---

## 2. Data flow

### Connect

```
HomeScreen._initialize()
  └─ BluetoothService.scanForGlasses()          [adapter, kept for the existing indicator]
       └─ GlassDeviceService.scanForGlasses()   MethodChannel "scan"
            └─ GlassBridge.bondedGlasses()      bonded BT Classic devices, name-filtered
       └─ GlassDeviceService.connect()          MethodChannel "connect"
            └─ GlassBridge.startConnect()
                 ├─ PeripheralConnectionManager.startConnect(SPP)
                 ├─ VendorModelCallback.onStateChanged(STATE_DATA_PREPARED)
                 └─ SmartWearModelClient.initSmartWearDevice()
                      └─ onSmartWearDeviceInitSuccess()
                           └─ EventChannel  {type: DEVICE_READY}
                           └─ EventChannel  {type: CONNECTION_STATE, state: connected}
```

`connect()` subscribes to `DEVICE_READY` **before** issuing the command. The
method channel returns as soon as the request is queued; the handshake lands
asynchronously, and a broadcast stream drops events that have no listener yet.
Getting this backwards produces an intermittent "connects but never reports
ready" bug that only shows on slower links.

### Capture — the one path both buttons take

```
 on-screen ElevatedButton ─┐
                           ├─→ _onCapturePressed() ─→ _capturePrescan() / _captureFinger()
 frame button ─────────────┘                              └─ CameraSourceController.captureJpeg()
   onDeviceTriggeredTakePhoto()                                 └─ GlassCameraSource  → SmartWearModelClient.startTakePhotoProcess(TYPE_TAKE_ORIGIN_IMAGE)
     └─ EventChannel {type: BUTTON_CLICKED}                     └─ PhoneCameraSource  → CameraController.takePicture()
          └─ GlassDeviceService.buttonClicks
```

Both controls converge on `_onCapturePressed` in each screen. They cannot
diverge, which was the point — a wearer using the frame button and a sighted
helper using the screen must get identical behaviour.

Stills are pulled at **full resolution over the vendor channel**, not grabbed
from the video feed. The live feed is downscaled; Braille dot detection needs
the original. See `CameraService`'s existing note on why `ResolutionPreset.medium`
was already too low for this pipeline.

### Audio

```
GlassDeviceService connect/disconnect
  └─ AudioRoutingService.syncToGlasses()
       ├─ AudioManager.mode = MODE_IN_COMMUNICATION
       ├─ startBluetoothSco()
       └─ waits for ACTION_SCO_AUDIO_STATE_UPDATED == CONNECTED
            └─ only now is the glasses mic what speech_to_text will open
```

The wait is the important part. `startBluetoothSco()` is asynchronous; if the
Dart side is told "routed" immediately, STT opens the built-in phone mic and the
wearer's speech is captured by the wrong device with no visible error.

Output follows A2DP automatically once the headset is connected — except while
SCO is up, when everything collapses onto the narrowband SCO link. That is
acceptable for prompts and is what puts them in the wearer's ear.

---

## 3. File map

### New — Dart (`braille_lens_flutter/lib/services/`)

**`glass_device_service.dart`** — owns the link.
- `GlassConnectionState`, `GlassDevice`, and a **sealed `GlassEvent` family**
  (`GlassButtonClicked`, `GlassMicData`, `GlassPhotoCaptured`,
  `GlassConnectionChanged`, `GlassBattery`, `GlassLiveStream`, …). Sealed means
  a `switch` over an event is exhaustive at compile time; adding a native event
  type without handling it is a build error, not a silent no-op.
- One internal broadcast `StreamController`. However many screens subscribe,
  the native side sees exactly **one** `onListen`/`onCancel` pair — repeated
  `receiveBroadcastStream()` calls would otherwise churn the bridge.
- `ChangeNotifier` for connection/battery state, so screens can use
  `ListenableBuilder`. **No Bloc/Riverpod dependency is introduced** — this
  matches the plain-service style already in `lib/services/`.
- Exponential-backoff auto-reconnect, capped at 30s, suppressed after an
  explicit `disconnect()`.
- Fail-soft: `MissingPluginException` → state `unavailable`, and the app falls
  back to phone hardware rather than crashing. Same convention as
  `PrescanBridge`.

**`audio_routing_service.dart`** — owns SCO/A2DP routing, described above.
Deliberately does **not** go through the Realtek SDK: the glasses present a
standard Bluetooth headset, so this keeps working even when the vendor channel
is down.

**`camera_source.dart`** — the hardware abstraction.
- `CameraSource` — `initialize()`, `captureJpeg()`, `buildPreview()`,
  `dispose()`, `label`, `lastError`.
- `PhoneCameraSource` — wraps the existing `CameraService` unchanged.
- `GlassCameraSource` — RTSP preview, full-resolution stills over the vendor
  channel.
- `CameraSourceController` — picks the source, and **swaps it when the glasses
  connect or drop**. Falls back to the phone if the glasses feed will not
  start, and does so *without resetting prescan state*: losing the glasses
  mid-session must not throw away a page the wearer already scanned.

### New — Android (`android/app/src/main/kotlin/.../`)

**`GlassBridge.kt`** — `SmartWearModelClient` behind
`braille_lens/glass_control` (MethodChannel) and `braille_lens/glass_events`
(EventChannel).
- SDK bring-up (`RtkCore`, `MultiPeripheralConnectionManager`,
  `SmartWearModelProxy`) is **lazy and one-shot**, so the Flutter manifest keeps
  the default Application class. The Realtek reference app does this in its
  `Application`; doing it on first use is equivalent because every entry point
  goes through `ensureSdk()`.
- Threading: SDK callbacks arrive on its own worker threads; everything
  outbound is posted to the main looper before touching a Flutter channel.
- `TransportLayer.TDBG = false` — verbose transport logging measurably slows
  image and voice transfer.

**`AudioRoutingBridge.kt`** — `AudioManager` + a receiver for
`ACTION_SCO_AUDIO_STATE_UPDATED` and `ACTION_CONNECTION_STATE_CHANGED`.

**`MainActivity.kt`** — registers both bridges, disposes both in `onDestroy()`.
Leaving either up keeps the glasses mic hot after the app is gone.

**`android/app/libs/`** — vendored Realtek AARs and JARs.

> **Do not add `rtk-core-*.jar`.** It duplicates every class in
> `rtk-core-ktx-*.jar` and D8 fails with hundreds of `Duplicate class
> com.realsil.sdk.core.*` errors. The reference app ships only the `-ktx`
> variant. Mirror its dependency list rather than adding every file in `libs/`.

### Modified

| File | Change |
| --- | --- |
| `lib/services/bluetooth_service.dart` | Mock scan deleted. Now a thin adapter over `GlassDeviceService` so `home_screen`'s indicator kept its shape. **New code should use `GlassDeviceService` directly** — it exposes battery, mic and live-feed state this adapter hides. |
| `lib/screens/learning_screen.dart` | Holds a `CameraSourceController` instead of a `CameraService`; frame button and on-screen button both route through `_onCapturePressed`; preview delegates to the source. |
| `lib/screens/testing_screen.dart` | Same treatment. |
| `lib/screens/home_screen.dart` | Requests API 31+ Bluetooth runtime permissions; syncs the audio route on connect; announces battery level. |
| `android/app/src/main/AndroidManifest.xml` | `BLUETOOTH_SCAN` / `BLUETOOTH_CONNECT` (plus pre-31 `BLUETOOTH` / `BLUETOOTH_ADMIN` and capped `ACCESS_FINE_LOCATION`), `MODIFY_AUDIO_SETTINGS`, and the Wi-Fi state permissions the RTSP path needs. |
| `android/app/build.gradle.kts` | Vendored SDK deps, their transitive AndroidX/Guava/OkHttp/Gson requirements, and packaging excludes for the duplicated `META-INF` entries. |

**Untouched on purpose:** the ONNX pipeline
(`prescan_bridge`, `classifier_service`, `fingertip_onnx_service`,
`covered_cell_service`, `coordinate_mapper`). Both camera sources hand back a
JPEG in the orientation that pipeline already expects, so the hardware swap is
invisible to it.

---

## 4. Permissions

Runtime permissions are requested in `HomeScreen._initialize()`. On API 31+,
**`BLUETOOTH_CONNECT` is required even to read the bonded-device list** — without
it `bondedGlasses()` returns empty and the app reports "no glasses found" with
no other symptom. That is the first thing to check when discovery fails on a
modern device.

---

## 5. Testing notes

- **Emulators cannot run this.** The Realtek AAR ships no x86_64 slice, and
  there is no Bluetooth Classic peripheral to talk to. Physical ARM device plus
  real glasses only.
- Pair the glasses in Android Bluetooth settings first. The bridge connects to
  *bonded* devices; it does not perform pairing itself.
- `flutter analyze` and a release build both pass; the release APK is ~63 MB.

---

## 6. Live preview

The bridge starts the session (`RTKLiveStreamingManager.startLiveStreaming`)
and, once the device reports it is up, emits:

```
{type: LIVE_STREAM, channel, ssid, password, ipAddress, rtspUrl: "rtsp://<ip>:554"}
```

`rtsp://<ip>:554` with **no path component** is exactly the URL the Realtek
reference app plays (`DebugLiveStreamingActivity`), so it is not a guess.

`GlassCameraSource` subscribes to `GlassDeviceService.liveStreams`, opens that
URL in a `media_kit` `Player`, and renders it with a `Video` widget. libmpv does
the H.264 depacketise and decode, so we ship no custom native decoder. The
player is opened with a small buffer — this is a viewfinder, so latency beats
smoothing over a dropped frame — and disposed *before* the device is told to
stop, so libmpv is never left reading a socket that is closing.

### Mode caveat

The two Wi-Fi modes behave differently, and it matters:

- **WIFI_STATION** — the glasses join a network and report a routable IP.
  `WifiAccessInfo.ipAddress` is populated, the URL is reachable, and the
  preview works.
- **WIFI_AP** — the glasses host their own SoftAP and the SDK's internal
  `RTKMediaPlayerService` handles the link; no URL is exposed to callers. An
  external player only works here if the phone has joined that SoftAP.

`GlassCameraSource` handles the empty-URL case explicitly rather than spinning:
it says the feed has no reachable address and notes that **capture still
works**, which is true — stills come over the vendor channel, not this feed.

### Build note

`media_kit_libs_android_video` downloads libmpv jars from GitHub during the
Gradle build into `build/media_kit_libs_android_video/v1.1.7/` and MD5-checks
them. On a flaky connection this fails with `Connection reset`. Pre-fetch them
with curl if needed — but **not** with `curl -C -` onto a partial file, which
produces a corrupt jar that fails the checksum. Delete and re-download whole.

The release APK is ~80 MB, up from ~63 MB before libmpv.

---

## 7. The frame button: no configuration required

An earlier draft of this document claimed the button needed "custom take photo
behaviour" enabled on the device. **That was wrong, and it is worth recording
why**, because the reference app makes it look true.

Disassembling `SmartWearModelClient` (`javap -c`) shows the dispatch:

```
action = payload[0] & 0xFF
log("received device action: " + action)
if (action == 1) -> onDeviceTriggeredTakePhoto()   // ACTION_TAKE_PHOTO
else             -> onReceivedDeviceAction(action)
```

It is unconditional. There is no gate. Supporting evidence:

- `SmartWearConfig.Builder` exposes only `setDeviceType`,
  `setOutputAudioSampleRate` and `setOutputAudioSampleChannel`, and
  `setSmartWearDeviceParam()` accepts nothing else.
- No class in the SDK contains the string `Behavior`.
- The reference app's `SETTINGS_CUSTOM_TAKE_PHOTO_BEHAVIOR` pref is written by
  `GlassSettingsActivity` and **read only to populate its own switch**. Nothing
  applies it to the device or the SDK. It is dead config in the sample.

So if the frame button appears not to work, check, in order: the callback is
registered (`SmartWearModelClient.registerCallback`), the SPP link is up, and
`BLUETOOTH_CONNECT` was granted. Do not go looking for a setting.

Note also that `onReceivedDeviceAction` never receives `ACTION_TAKE_PHOTO` —
handling it there as well would be dead code, which is why `GlassBridge` only
logs from that callback.

---

## 8. Remaining work

- Verify on hardware: button, mic routing, and RTSP preview in WIFI_STATION
  mode have not yet been exercised against real glasses.
- `GlassVideoFrame` / `onReceivedLiveStreamingData` (the BT live channel) is
  modelled in Dart but not wired; only the Wi-Fi RTSP path is implemented.
