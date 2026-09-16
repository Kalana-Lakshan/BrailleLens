package com.braillelens.braille_lens_flutter

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.realsil.sdk.audioconnect.smartwear.FileStorageInfo
import com.realsil.sdk.audioconnect.smartwear.SmartWearConstants
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelCallback
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelClient
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelProxy
import com.realsil.sdk.audioconnect.smartwear.WifiApInfo
import com.realsil.sdk.audioconnect.smartwear.config.LiveStreamingConfigInfo
import com.realsil.sdk.audioconnect.smartwear.entity.ThumbnailPhoto
import com.realsil.sdk.audioconnect.smartwear.entity.WifiAccessInfo
import com.realsil.sdk.bbpro.BeeProParams
import com.realsil.sdk.bbpro.MultiPeripheralConnectionManager
import com.realsil.sdk.bbpro.PeripheralConnectionManager
import com.realsil.sdk.bbpro.core.peripheral.ConnectionParameters
import com.realsil.sdk.bbpro.core.peripheral.PeripheralParameters
import com.realsil.sdk.bbpro.core.transportlayer.TransportLayer
import com.realsil.sdk.bbpro.vendor.VendorModelCallback
import com.realsil.sdk.core.RtkConfigure
import com.realsil.sdk.core.RtkCore
import com.realsil.sdk.core.logger.ZLogger
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Bridges the Realtek Audio Connect SDK to Dart's `GlassDeviceService`.
 *
 * The AI Glass is a Bluetooth **Classic** peripheral speaking Realtek's vendor
 * protocol over SPP — not a BLE GATT device — so the frame button, microphone
 * PCM and photo transfer all arrive on [SmartWearModelCallback] rather than on
 * GATT characteristics. That is why this lives in Kotlin instead of being done
 * with `flutter_blue_plus` in Dart.
 *
 * Threading: SDK callbacks land on its own worker threads, while Flutter
 * channels must be touched from the main thread. Everything outbound is
 * therefore posted through [main].
 */
class GlassBridge(context: Context, messenger: BinaryMessenger) :
    MethodChannel.MethodCallHandler, EventChannel.StreamHandler {

    companion object {
        private const val TAG = "GlassBridge"
        private const val CONTROL_CHANNEL = "braille_lens/glass_control"
        private const val EVENT_CHANNEL = "braille_lens/glass_events"

        /** Realtek's SPP service UUID prefix, used to spot glasses among bonded devices. */
        private const val GLASS_NAME_HINT = "glass"

        @Volatile
        private var sdkInitialised = false
    }

    private val appContext: Context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())

    private val methodChannel = MethodChannel(messenger, CONTROL_CHANNEL).also {
        it.setMethodCallHandler(this)
    }
    private val eventChannel = EventChannel(messenger, EVENT_CHANNEL).also {
        it.setStreamHandler(this)
    }

    private var events: EventChannel.EventSink? = null
    private var connectionManager: PeripheralConnectionManager? = null
    private var modelClient: SmartWearModelClient? = null
    private var deviceAddress: String? = null

    /** Channel the current live stream was started on; stopLiveStreaming needs it. */
    private var activeLiveChannel: Byte = 0

    // ── Event plumbing ─────────────────────────────────────────────────────────

    override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) {
        events = sink
    }

    override fun onCancel(arguments: Any?) {
        events = null
    }

    private fun emit(vararg pairs: Pair<String, Any?>) {
        val map = hashMapOf<String, Any?>(*pairs)
        main.post { events?.success(map) }
    }

    private fun emitError(message: String, code: String) =
        emit("type" to "ERROR", "message" to message, "code" to code)

    private fun emitConnectionState(state: String) = emit(
        "type" to "CONNECTION_STATE",
        "state" to state,
        "address" to deviceAddress,
        "name" to (bondedName(deviceAddress) ?: "AI Glass"),
    )

    // ── SDK lifecycle ──────────────────────────────────────────────────────────

    /**
     * One-shot SDK bring-up. The reference app does this in its Application
     * class; doing it lazily here keeps the Flutter manifest untouched and is
     * equivalent, since every entry point below goes through [ensureSdk].
     */
    private fun ensureSdk(): Boolean {
        if (sdkInitialised) return true
        return try {
            val configure = RtkConfigure.Builder()
                .debugEnabled(false)
                .printLog(false)
                .globalLogLevel(ZLogger.WARN)
                .logTag("BrailleLensGlass")
                .devModeEnabled(false)
                .build()
            RtkCore.initialize(appContext, configure)

            val params = BeeProParams.Builder()
                .syncDataWhenConnected(true)
                .connectA2dp(true)   // prompt/TTS playback through the glasses
                .listenHfp(true)     // SCO mic
                .build()
            MultiPeripheralConnectionManager.getInstance(appContext).initialize(params)
            SmartWearModelProxy.initialize(appContext)

            // Verbose transport logging measurably slows image and voice
            // transfer; the reference app disables it for release too.
            TransportLayer.TDBG = false

            sdkInitialised = true
            true
        } catch (t: Throwable) {
            Log.e(TAG, "SDK init failed", t)
            emitError("Realtek SDK init failed: ${t.message}", "SDK_INIT")
            false
        }
    }

    @SuppressLint("MissingPermission")
    private fun bondedGlasses(): List<BluetoothDevice> {
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return emptyList()
        if (!adapter.isEnabled) return emptyList()
        return try {
            adapter.bondedDevices.orEmpty().filter {
                val n = it.name?.lowercase().orEmpty()
                n.contains(GLASS_NAME_HINT) || n.contains("rtk") || n.contains("aiglass")
            }
        } catch (se: SecurityException) {
            Log.w(TAG, "BLUETOOTH_CONNECT not granted", se)
            emptyList()
        }
    }

    @SuppressLint("MissingPermission")
    private fun bondedName(address: String?): String? {
        if (address == null) return null
        return try {
            BluetoothAdapter.getDefaultAdapter()
                ?.bondedDevices
                ?.firstOrNull { it.address == address }
                ?.name
        } catch (se: SecurityException) {
            null
        }
    }

    // ── Method channel ─────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "initialize" -> result.success(ensureSdk())

            "scan" -> {
                if (!ensureSdk()) return result.success(emptyList<Any>())
                val found = bondedGlasses().map {
                    hashMapOf<String, Any?>(
                        "address" to it.address,
                        "name" to (it.name ?: "AI Glass"),
                    )
                }
                found.forEach {
                    emit(
                        "type" to "SCAN_RESULT",
                        "address" to it["address"],
                        "name" to it["name"],
                    )
                }
                result.success(found)
            }

            "connect" -> {
                if (!ensureSdk()) return result.success(false)
                val address = call.argument<String>("address")
                    ?: deviceAddress
                    ?: bondedGlasses().firstOrNull()?.address
                if (address == null) {
                    emitError("No bonded AI Glass found — pair it in Android settings first", "NO_DEVICE")
                    return result.success(false)
                }
                result.success(startConnect(address))
            }

            "disconnect" -> {
                teardown()
                emitConnectionState("disconnected")
                result.success(null)
            }

            "takePhoto" -> {
                val client = modelClient
                    ?: return result.success(false).also {
                        emitError("Glasses not connected", "NOT_CONNECTED")
                    }
                // Full-resolution original: the thumbnail stream is far too
                // small for Braille dot detection.
                val err = client.startTakePhotoProcess(
                    SmartWearConstants.TakePhotoType.TYPE_TAKE_ORIGIN_IMAGE
                )
                result.success(err != null && err.code == 0)
            }

            "startMic" -> {
                val client = modelClient
                    ?: return result.success(false).also {
                        emitError("Glasses not connected", "NOT_CONNECTED")
                    }
                val err = client.startUserVoiceInput()
                result.success(err != null && err.code == 0)
            }

            "stopMic" -> {
                val err = modelClient?.stopUserVoiceInput()
                result.success(err != null && err.code == 0)
            }

            "startLiveStream" -> result.success(startLiveStream(call.argument<Int>("channel")))

            "stopLiveStream" -> {
                runCatching {
                    modelClient?.liveStreamingManager?.stopLiveStreaming(activeLiveChannel)
                }
                activeLiveChannel = 0
                result.success(true)
            }

            "getBattery" -> {
                modelClient?.deviceBatteryInfo
                result.success(null)
            }

            else -> result.notImplemented()
        }
    }

    // ── Connection ─────────────────────────────────────────────────────────────

    private fun startConnect(address: String): Boolean = try {
        teardown()
        deviceAddress = address
        emitConnectionState("connecting")

        val manager = MultiPeripheralConnectionManager.getInstance(appContext)
            .getPeripheralConnectionManager(address)
        connectionManager = manager
        manager?.registerVendorModelCallback(vendorCallback)

        modelClient = SmartWearModelProxy.getInstance().getModelClient(address)?.also {
            it.registerCallback(modelCallback)
        }

        if (manager?.isConnected == true) {
            // Already linked from a previous session — go straight to the
            // SmartWear handshake rather than re-running SPP setup.
            onLinkReady()
        } else {
            val peripheralParams = PeripheralParameters.Builder()
                .syncDataWhenConnected(true)
                .connectA2dp(true)
                .listenHfp(true)
                .build()
            val connectParams = ConnectionParameters.Builder(address)
                .channelType(ConnectionParameters.CHANNEL_TYPE_SPP)
                .peripheralParameters(peripheralParams)
                .build()
            manager?.startConnect(connectParams)
        }
        true
    } catch (t: Throwable) {
        Log.e(TAG, "connect failed", t)
        emitError("Connect failed: ${t.message}", "CONNECT")
        emitConnectionState("disconnected")
        false
    }

    /**
     * SPP is up. Ask the SmartWear layer to initialise; Dart is told the device
     * is ready only once [SmartWearModelCallback.onSmartWearDeviceInitSuccess]
     * lands, because the vendor channel is not usable before that.
     */
    private fun onLinkReady() {
        modelClient?.let {
            it.registerCallback(modelCallback)
            it.initSmartWearDevice()
        } ?: run {
            modelClient = SmartWearModelProxy.getInstance()
                .getModelClient(deviceAddress)
                ?.also { c ->
                    c.registerCallback(modelCallback)
                    c.initSmartWearDevice()
                }
        }
    }

    private fun teardown() {
        runCatching { connectionManager?.unregisterVendorModelCallback(vendorCallback) }
        runCatching { modelClient?.unregisterCallback(modelCallback) }
        connectionManager = null
        modelClient = null
    }

    private val vendorCallback = object : VendorModelCallback() {
        override fun onStateChanged(state: Int) {
            super.onStateChanged(state)
            when (state) {
                PeripheralConnectionManager.STATE_DATA_PREPARED -> onLinkReady()
                PeripheralConnectionManager.STATE_DEVICE_DISCONNECTED -> {
                    emitConnectionState("disconnected")
                }
            }
        }
    }

    // ── Live streaming ─────────────────────────────────────────────────────────

    private fun startLiveStream(channel: Int?): Boolean {
        val client = modelClient ?: run {
            emitError("Glasses not connected", "NOT_CONNECTED")
            return false
        }
        return try {
            activeLiveChannel = (channel ?: 1).toByte()
            val config = LiveStreamingConfigInfo().apply {
                liveStreamingChannel = when (channel) {
                    2 -> LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_STATION
                    3 -> LiveStreamingConfigInfo.LiveStreamingChannel.BT
                    else -> LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_AP
                }
                videoEncoding = LiveStreamingConfigInfo.VideoEncoding.H264
            }
            client.liveStreamingManager?.startLiveStreaming(config)
            true
        } catch (t: Throwable) {
            Log.e(TAG, "live stream failed", t)
            emitError("Live stream failed: ${t.message}", "LIVE_STREAM")
            false
        }
    }

    // ── SmartWear callbacks ────────────────────────────────────────────────────

    private val modelCallback = object : SmartWearModelCallback() {

        override fun onSmartWearDeviceInitSuccess() {
            emit("type" to "DEVICE_READY", "success" to true)
            emitConnectionState("connected")
            runCatching { modelClient?.deviceBatteryInfo }
        }

        override fun onSmartWearDeviceInitFail() {
            emit("type" to "DEVICE_READY", "success" to false)
            emitConnectionState("disconnected")
        }

        /**
         * The physical frame button. This is the event Dart maps onto the same
         * handler as the on-screen capture button.
         *
         * Fires unconditionally: the SDK reads the action byte off the vendor
         * packet and dispatches ACTION_TAKE_PHOTO (1) straight here, routing
         * every *other* action value to onReceivedDeviceAction instead. There
         * is no config flag gating it — SmartWearConfig carries only device
         * type and audio format, and setSmartWearDeviceParam takes nothing
         * else. So if the button appears dead, the cause is upstream (callback
         * not registered, or the link down), not a missing setting.
         */
        override fun onDeviceTriggeredTakePhoto() {
            emit("type" to "BUTTON_CLICKED")
        }

        /**
         * Every device action *except* ACTION_TAKE_PHOTO, which the SDK
         * delivers through onDeviceTriggeredTakePhoto above. Surfaced so an
         * unexpected action shows up in logs rather than vanishing.
         */
        override fun onReceivedDeviceAction(action: Int) {
            Log.d(TAG, "device action $action")
        }

        override fun onStartReceiveUserVoice() =
            emit("type" to "MIC_STATE", "streaming" to true)

        override fun onStopReceiveUserVoice() =
            emit("type" to "MIC_STATE", "streaming" to false)

        /** 16 kHz mono 16-bit PCM — already the shape the STT model wants. */
        override fun onReceivedUserVoice(pcmData: ByteArray) =
            emit("type" to "MIC_DATA", "data" to pcmData)

        override fun onTakePhotoSuccess(photos: MutableList<ThumbnailPhoto>) {
            val file = photos.firstOrNull()?.photoFile
            if (file == null || !file.exists()) {
                emit("type" to "PHOTO_FAILED", "reason" to "no file returned")
                return
            }
            try {
                emit("type" to "PHOTO_CAPTURED", "data" to file.readBytes())
            } catch (t: Throwable) {
                emit("type" to "PHOTO_FAILED", "reason" to (t.message ?: "read failed"))
            }
        }

        override fun onTakePhotoFail(reason: Int) =
            emit("type" to "PHOTO_FAILED", "reason" to "code $reason")

        override fun onReceivedDeviceBatteryInfo(level: Int, status: Int) =
            emit("type" to "BATTERY", "level" to level, "charging" to (status == 1))

        override fun onWifiApStateChanged(info: WifiApInfo?) = emit(
            "type" to "WIFI_AP_STATE",
            "ssid" to info?.ssid,
            "password" to info?.password,
            "mode" to (info?.mode?.toInt() ?: 0),
        )

        override fun onStartLiveStreaming(
            channel: Byte,
            success: Boolean,
            access: WifiAccessInfo?,
        ) {
            if (!success) {
                emitError("Glasses refused the live stream", "LIVE_STREAM")
                return
            }
            // Same URL shape the Realtek reference app plays:
            // rtsp://<device ip>:554, no path component.
            val ip = access?.ipAddress
            emit(
                "type" to "LIVE_STREAM",
                "channel" to channel.toInt(),
                "ssid" to access?.ssid,
                "password" to access?.password,
                "ipAddress" to ip,
                "rtspUrl" to if (ip.isNullOrEmpty()) "" else "rtsp://$ip:554",
            )
        }

        override fun onFileStorageInfoChanged(info: FileStorageInfo?) {
            // Not surfaced to Dart yet — BrailleLens pulls stills straight over
            // the vendor channel rather than browsing on-glasses storage.
        }
    }

    fun dispose() {
        teardown()
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        events = null
    }
}
