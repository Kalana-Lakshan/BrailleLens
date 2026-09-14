package com.braillelens.braille_lens_flutter

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

import com.realsil.sdk.audioconnect.ai.realtime.RealtimeApiConfig
import com.realsil.sdk.audioconnect.ai.realtime.RealtimeApiConstants
import com.realsil.sdk.audioconnect.smartwear.SmartWearConfig
import com.realsil.sdk.audioconnect.smartwear.SmartWearConstants
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelCallback
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelClient
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelProxy
import com.realsil.sdk.audioconnect.smartwear.WifiApInfo
import com.realsil.sdk.audioconnect.smartwear.config.LiveStreamingConfigInfo
import com.realsil.sdk.audioconnect.smartwear.entity.WifiAccessInfo

/**
 * Native bridge between the BrailleLens Flutter layer and the Realtek smart glasses.
 *
 * The SDK itself is brought up in [MainApplication]; this class only owns the
 * per-device client and the two Flutter channels.
 *
 * Control ([METHOD_CHANNEL]):
 *   `hasPermissions` / `requestPermissions`
 *   `initDevice{btAddress}`  bind a client, configure mic format and device type
 *   `startMicStream` / `stopMicStream`
 *   `startCameraStream{channel}` / `stopCameraStream{channel}`
 *   `release`
 *
 * Events ([EVENT_CHANNEL]) — every payload is a map with a `type` key:
 *   `DEVICE_READY`     `success`: result of the async initSmartWearDevice handshake
 *   `MIC_DATA`         `data`: 16 kHz mono PCM chunk (arrives as Uint8List in Dart)
 *   `MIC_STATE`        `streaming`: Bool
 *   `BUTTON_CLICKED`   the frame's hardware button (SDK action ACTION_TAKE_PHOTO)
 *   `LIVE_STREAM`      `ssid`, `password`, `ipAddress`, `rtspUrl`, `channel`
 *   `WIFI_AP_STATE`    `ssid`, `password` — credentials of the glasses' own AP
 *   `VIDEO_DATA`       `data`: H.264 bitstream chunk, BT channel only
 *   `ERROR`            `message`
 *
 * SDK callbacks arrive on the SDK's worker threads; [emit] marshals every one of
 * them onto the main looper before touching the Flutter sink, which is not
 * thread-safe.
 */
class MainActivity : FlutterActivity() {

    companion object {
        private const val TAG = "BrailleLensGlasses"

        private const val METHOD_CHANNEL = "braille_lens/glasses_control"
        private const val EVENT_CHANNEL = "braille_lens/glasses_events"

        /**
         * Fallback RTSP endpoint, used when the device does not report its IP.
         * 192.168.43.1 is the gateway the glasses hand out in Wi-Fi AP mode.
         */
        private const val DEFAULT_RTSP_URL = "rtsp://192.168.43.1:554"
        private const val RTSP_PORT = 554

        private const val REQ_GLASSES_PERMISSIONS = 9001
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    private var methodChannel: MethodChannel? = null
    private var eventChannel: EventChannel? = null

    /** Main-thread only: assigned in onListen/onCancel, read in [emit]. */
    private var eventSink: EventChannel.EventSink? = null

    private var modelClient: SmartWearModelClient? = null
    private var boundAddress: String? = null

    /** Channel the current live stream was started on, needed to stop it again. */
    private var activeLiveChannel: Byte? = null

    /** Pending result for an in-flight permission round trip. */
    private var permissionResult: MethodChannel.Result? = null

    // -----------------------------------------------------------------------
    // Flutter engine wiring
    // -----------------------------------------------------------------------

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        eventChannel = EventChannel(messenger, EVENT_CHANNEL).apply {
            setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    eventSink = events
                }

                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            })
        }

        methodChannel = MethodChannel(messenger, METHOD_CHANNEL).apply {
            setMethodCallHandler { call, result -> handleMethodCall(call, result) }
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        releaseClient()
        methodChannel?.setMethodCallHandler(null)
        eventChannel?.setStreamHandler(null)
        methodChannel = null
        eventChannel = null
        eventSink = null
        super.cleanUpFlutterEngine(flutterEngine)
    }

    override fun onDestroy() {
        releaseClient()
        mainHandler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    // -----------------------------------------------------------------------
    // MethodChannel
    // -----------------------------------------------------------------------

    private fun handleMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            when (call.method) {
                "hasPermissions" -> result.success(hasGlassesPermissions())

                "requestPermissions" -> requestGlassesPermissions(result)

                "initDevice" -> {
                    val address = call.argument<String>("btAddress")
                    if (address.isNullOrBlank()) {
                        result.error("BAD_ARGS", "btAddress is required", null)
                    } else {
                        initDevice(address, result)
                    }
                }

                "startMicStream" -> withClient(result) { client ->
                    client.startUserVoiceInput()
                    result.success(true)
                }

                "stopMicStream" -> withClient(result) { client ->
                    client.stopUserVoiceInput()
                    result.success(true)
                }

                "startCameraStream" -> withClient(result) { client ->
                    val channel = (call.argument<Int>("channel") ?: LIVE_CHANNEL_WIFI_AP.toInt()).toByte()
                    startLiveStreaming(client, channel)
                    // Resolves as soon as the command is queued; readiness and the
                    // stream credentials arrive later as a LIVE_STREAM event.
                    result.success(true)
                }

                "stopCameraStream" -> withClient(result) { client ->
                    val channel = (call.argument<Int>("channel"))?.toByte()
                        ?: activeLiveChannel
                        ?: LIVE_CHANNEL_WIFI_AP
                    client.liveStreamingManager.stopLiveStreaming(channel)
                    client.liveStreamingManager.release()
                    activeLiveChannel = null
                    result.success(true)
                }

                "release" -> {
                    releaseClient()
                    result.success(true)
                }

                else -> result.notImplemented()
            }
        } catch (t: Throwable) {
            Log.e(TAG, "method ${call.method} failed", t)
            result.error("SDK_ERROR", t.message ?: t.javaClass.simpleName, null)
        }
    }

    /** Runs [block] with a bound client, or fails the call if there isn't one. */
    private inline fun withClient(
        result: MethodChannel.Result,
        block: (SmartWearModelClient) -> Unit,
    ) {
        val client = modelClient
        if (client == null) {
            result.error("NOT_CONNECTED", "Call initDevice(btAddress) first", null)
            return
        }
        block(client)
    }

    // -----------------------------------------------------------------------
    // Device setup
    // -----------------------------------------------------------------------

    private fun initDevice(btAddress: String, result: MethodChannel.Result) {
        if (!hasGlassesPermissions()) {
            result.error("NO_PERMISSION", "Bluetooth permissions not granted", null)
            return
        }

        // Re-binding to a different pair of glasses: drop the previous client first.
        if (boundAddress != null && boundAddress != btAddress) releaseClient()

        val client = SmartWearModelProxy.getInstance().getModelClient(btAddress)
        if (client == null) {
            result.error(
                "NO_CLIENT",
                "No SmartWear client for $btAddress — is the device paired and connected?",
                null,
            )
            return
        }

        client.registerCallback(modelCallback)

        // Identify as smart glasses, and keep the shutter button for ourselves:
        // setUseCustomTakePhotoBehavior(true) stops the SDK running its built-in
        // capture flow so the press surfaces as onReceivedDeviceAction and we can
        // drive the BrailleLens vision pipeline instead.
        client.setAIModelParam(
            RealtimeApiConfig.Builder()
                .setSmartDeviceType(RealtimeApiConstants.DeviceType.TYPE_SMART_GLASS)
                .setUseCustomTakePhotoBehavior(true)
                .build()
        )

        // 16 kHz mono PCM — the format the offline STT model expects, so the stream
        // needs no resampling on its way into the ONNX pipeline.
        client.setSmartWearDeviceParam(
            SmartWearConfig.Builder()
                .setOutputAudioSampleRate(SmartWearConstants.AudioSampleRate.SAMPLE_RATE_16000)
                .setOutputAudioSampleChannel(SmartWearConstants.AudioChannel.CHANNEL_MONO)
                .build()
        )

        modelClient = client
        boundAddress = btAddress

        // Asynchronous handshake: the outcome lands on onSmartWearDeviceInitSuccess
        // or onSmartWearDeviceInitFail, re-emitted to Dart as DEVICE_READY.
        client.initSmartWearDevice()

        Log.i(TAG, "SmartWear client bound for $btAddress, awaiting init handshake")
        result.success(true)
    }

    private fun startLiveStreaming(client: SmartWearModelClient, channel: Byte) {
        val config = LiveStreamingConfigInfo().apply {
            liveStreamingChannel = channel
            videoEncoding = LiveStreamingConfigInfo.VideoEncoding.H264
            videoPictureWidth = 1280
            videoPictureHeight = 720
            fps = 30
            bps = 1_000_000
        }
        activeLiveChannel = channel
        client.liveStreamingManager.startLiveStreaming(config)
    }

    private fun releaseClient() {
        val client = modelClient ?: return
        try {
            client.unregisterCallback(modelCallback)
            client.stopUserVoiceInput()
            activeLiveChannel?.let { client.liveStreamingManager.stopLiveStreaming(it) }
            client.liveStreamingManager.release()
        } catch (t: Throwable) {
            Log.w(TAG, "error while releasing SmartWear client", t)
        } finally {
            modelClient = null
            boundAddress = null
            activeLiveChannel = null
        }
    }

    // -----------------------------------------------------------------------
    // SDK callbacks -> Flutter
    // -----------------------------------------------------------------------

    private val modelCallback = object : SmartWearModelCallback() {

        override fun onSmartWearDeviceInitSuccess() {
            emit(mapOf("type" to "DEVICE_READY", "success" to true))
        }

        override fun onSmartWearDeviceInitFail() {
            emit(mapOf("type" to "DEVICE_READY", "success" to false))
        }

        override fun onStartReceiveUserVoice() {
            emit(mapOf("type" to "MIC_STATE", "streaming" to true))
        }

        override fun onStopReceiveUserVoice() {
            emit(mapOf("type" to "MIC_STATE", "streaming" to false))
        }

        /** Microphone PCM from the glasses, on an SDK worker thread at ~32 KB/s. */
        override fun onReceivedUserVoice(voiceData: ByteArray) {
            // Copy: the SDK may recycle this buffer before the main-thread post runs,
            // and nothing is serialised until then.
            emit(mapOf("type" to "MIC_DATA", "data" to voiceData.copyOf()))
        }

        /**
         * Hardware controls on the frame.
         *
         * The parameter is an `int`, while SmartWearConstants.DeviceAction declares
         * `byte` constants, hence the widening comparison.
         */
        override fun onReceivedDeviceAction(actionType: Int) {
            if (actionType == SmartWearConstants.DeviceAction.ACTION_TAKE_PHOTO.toInt()) {
                emit(mapOf("type" to "BUTTON_CLICKED"))
            } else {
                Log.d(TAG, "unhandled device action: $actionType")
            }
        }

        /** Credentials of the AP the glasses themselves host. */
        override fun onWifiApStateChanged(wifiApInfo: WifiApInfo) {
            emit(
                mapOf(
                    "type" to "WIFI_AP_STATE",
                    "ssid" to wifiApInfo.ssid,
                    "password" to wifiApInfo.password,
                    "mode" to wifiApInfo.mode.toInt(),
                )
            )
        }

        /**
         * Result of [startLiveStreaming]. `startResult` is the readiness flag, not
         * a status code, and `wifiAccessInfo` is documented as only meaningful on
         * the WIFI_STATION channel.
         */
        override fun onStartLiveStreaming(
            liveChannel: Byte,
            startResult: Boolean,
            wifiAccessInfo: WifiAccessInfo?,
        ) {
            if (!startResult) {
                Log.e(TAG, "live streaming failed on channel $liveChannel")
                emit(
                    mapOf(
                        "type" to "ERROR",
                        "message" to "Live streaming failed on channel $liveChannel",
                    )
                )
                return
            }

            // getSSID()/getIpAddress() are called explicitly: the JavaBean names are
            // all-caps, so Kotlin's synthetic property for getSSID() is `SSID`.
            val ip = wifiAccessInfo?.getIpAddress()
            emit(
                mapOf(
                    "type" to "LIVE_STREAM",
                    "channel" to liveChannel.toInt(),
                    "ssid" to wifiAccessInfo?.getSSID(),
                    "password" to wifiAccessInfo?.getPassword(),
                    "ipAddress" to ip,
                    "rtspUrl" to if (ip.isNullOrBlank()) DEFAULT_RTSP_URL else "rtsp://$ip:$RTSP_PORT",
                )
            )
        }

        /**
         * Video bitstream over Bluetooth. Only fires on the BT channel, where the
         * glasses push H.264 directly and no Wi-Fi association is needed.
         */
        override fun onReceivedLiveStreamingData(streamingData: ByteArray) {
            emit(mapOf("type" to "VIDEO_DATA", "data" to streamingData.copyOf()))
        }
    }

    /**
     * Hops to the main looper and forwards [event] to Dart.
     *
     * `EventSink` must only be touched on the platform thread, and is null whenever
     * Dart has no active subscription (before `listen`, or after cancellation).
     */
    private fun emit(event: Map<String, Any?>) {
        mainHandler.post {
            val sink = eventSink
            if (sink == null) {
                Log.d(TAG, "dropping ${event["type"]} - no Dart listener")
                return@post
            }
            sink.success(event)
        }
    }

    // -----------------------------------------------------------------------
    // Permissions
    // -----------------------------------------------------------------------

    private fun requiredPermissions(): Array<String> = buildList {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            add(Manifest.permission.BLUETOOTH_SCAN)
            add(Manifest.permission.BLUETOOTH_CONNECT)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Required from Android 13 to associate with the glasses' Wi-Fi AP.
            add(Manifest.permission.NEARBY_WIFI_DEVICES)
        }
        // Still needed on 31+: the SDK's pairing flow scans for nearby devices.
        add(Manifest.permission.ACCESS_FINE_LOCATION)
    }.toTypedArray()

    private fun hasGlassesPermissions(): Boolean = requiredPermissions().all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestGlassesPermissions(result: MethodChannel.Result) {
        if (hasGlassesPermissions()) {
            result.success(true)
            return
        }
        if (permissionResult != null) {
            result.error("BUSY", "A permission request is already in flight", null)
            return
        }
        permissionResult = result
        ActivityCompat.requestPermissions(this, requiredPermissions(), REQ_GLASSES_PERMISSIONS)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        if (requestCode == REQ_GLASSES_PERMISSIONS) {
            // Clear the pending result before replying: the dialog can be dismissed,
            // and a Result may only ever be completed once.
            val pending = permissionResult
            permissionResult = null
            val granted = grantResults.isNotEmpty() &&
                grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            pending?.success(granted)
            return
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
}

/** Glasses host the AP; the phone joins it. See LiveStreamingConfigInfo.LiveStreamingChannel. */
private val LIVE_CHANNEL_WIFI_AP: Byte = LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_AP
