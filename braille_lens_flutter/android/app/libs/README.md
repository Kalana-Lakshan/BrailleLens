# Realtek Audio Connect SDK

Proprietary vendor binaries, picked up by the `fileTree` dependency in
`android/app/build.gradle.kts`. Copied from the vendor's AIGlass reference app
(`android/app/src/AIGlass/libs/`).

| Artifact | Provides |
| --- | --- |
| `rtk-audioconnect-smartwear-1.8.41.aar` | `…audioconnect.smartwear` + `…audioconnect.ai` — model client, callbacks, live streaming |
| `rtk-audioconnect-common-1.15.28.aar` | `com.realsil.sdk.bbpro` — `BeeProParams`, `MultiPeripheralConnectionManager`, `ModelClient` |
| `rtk-audioconnect-core-1.9.10.jar` | `com.realsil.sdk.bbpro.core` — `BeeError`, `TransportLayer` |
| `rtk-core-ktx-1.7.83.jar` | `com.realsil.sdk.core` — `RtkCore`, `RtkConfigure`, `ZLogger` |
| `rtk-androidx-core-1.0.26.jar` | `com.realtek.androidx.core` — support classes the above depend on |

## Deliberately not bundled

- **`rtk-core-1.7.30.jar`** — same packages as `rtk-core-ktx`; including both
  trips `checkDebugDuplicateClasses`. The reference app also ships only the ktx one.
- **`rtk-dfu-3.14.36.jar`** — firmware OTA. Add it only if we implement updates.
- **`rtk-support-*.aar`, `rtk-support-debugger-*.aar`** — the vendor demo's
  `BaseActivity`/scanner UI. Not usable from a Flutter host.
- **`bilibili-accountoauth-*.aar`** — live-platform OAuth, irrelevant here.

`proguard-rules.pro` keeps `com.realsil.**` and silences the references the
bundled AARs make to the omitted DFU/support artifacts.

## API notes

Verified by `javap` against these artifacts (1.8.41 / 1.15.28), not from docs.
Several points differ from the original integration notes:

- `BeeProParams` and `MultiPeripheralConnectionManager` are in
  **`com.realsil.sdk.bbpro`**, not `…audioconnect.pro`.
- `BeeProParams.Builder` has **no** `autoConnectOnStart()` or
  `functionModuleEnabled()`. Available: `serverEnabled`, `listenA2dp`,
  `listenHfp`, `bindHfpDisconnection`, `connectA2dp`, `syncDataWhenConnected`,
  `uuid`, `transport`.
- **`RtkCore.initialize(context, RtkConfigure)` is mandatory and must run first.**
- Configs are applied with `client.setAIModelParam(RealtimeApiConfig)` and
  `client.setSmartWearDeviceParam(SmartWearConfig)`.
- `onStartLiveStreaming(liveChannel: Byte, startResult: Boolean, wifiAccessInfo: WifiAccessInfo?)`
  — `startResult` is a readiness flag, not a status code.
- Live-stream constants are nested in `LiveStreamingConfigInfo`
  (`LiveStreamingChannel.WIFI_AP` = 1, `VideoEncoding.H264` = 1), all `byte`.
- `WifiAccessInfo` exposes `getSSID()` / `getPassword()` / `getIpAddress()`.
  The all-caps bean name means Kotlin's synthetic property is `SSID`, so the
  getters are called explicitly in `MainActivity.kt`.
- `onReceivedDeviceAction(int)` — the frame button reports
  `SmartWearConstants.DeviceAction.ACTION_TAKE_PHOTO` (byte `1`).

## Wi-Fi / RTSP caveat

The channels do not mean what the integration notes assumed:

- `WIFI_AP` (1) — **the glasses host the AP**; the phone must join it first.
  The reference app uses this channel for RTMP push, and its own javadoc says
  `wifiAccessInfo` "is only useful when the live stream type is WIFI_STATION",
  so SSID/password may arrive null here. The glasses' own AP credentials come
  from the `onWifiApStateChanged(WifiApInfo)` callback instead.
- `WIFI_STATION` (2) — **the phone hosts a soft AP** and the glasses join it.
  This is the path where the reference app builds an RTSP URL, as
  `rtsp://${wifiAccessInfo.ipAddress}:554`.
- `BT` (3) — H.264 arrives over Bluetooth via `onReceivedLiveStreamingData`,
  with no Wi-Fi association at all.

`MainActivity.kt` defaults to `WIFI_AP` as specified and falls back to the fixed
`rtsp://192.168.43.1:554` when no IP is reported, but the channel is selectable
per call. Which channel actually yields an RTSP feed needs confirming on hardware.
