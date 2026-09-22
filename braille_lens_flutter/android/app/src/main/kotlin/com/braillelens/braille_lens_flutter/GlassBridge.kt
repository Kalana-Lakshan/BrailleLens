package com.braillelens.braille_lens_flutter

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothProfile
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import com.realsil.sdk.audioconnect.ai.realtime.RealtimeApiConfig
import com.realsil.sdk.audioconnect.ai.realtime.RealtimeApiConstants
import com.realsil.sdk.audioconnect.smartwear.FileStorageInfo
import com.realsil.sdk.audioconnect.smartwear.SmartWearConfig
import com.realsil.sdk.audioconnect.smartwear.SmartWearConstants
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelCallback
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelClient
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelProxy
import com.realsil.sdk.audioconnect.smartwear.WifiApInfo
import com.realsil.sdk.audioconnect.smartwear.WifiStaInfo
import com.realsil.sdk.audioconnect.smartwear.config.LiveStreamingConfigInfo
import com.realsil.sdk.audioconnect.smartwear.config.ThumbnailPhotoConfigInfo
import com.realsil.sdk.audioconnect.smartwear.entity.ThumbnailPhoto
import com.realsil.sdk.audioconnect.smartwear.entity.WifiAccessInfo
import com.realsil.sdk.audioconnect.smartwear.image.ImageCorrectCallback
import com.realsil.sdk.audioconnect.smartwear.image.RTKImageToolkit
import com.realsil.sdk.bbpro.BeeProParams
import com.realsil.sdk.bbpro.MultiPeripheralConnectionManager
import com.realsil.sdk.bbpro.PeripheralConnectionManager
import com.realsil.sdk.bbpro.core.BeeError
import com.realsil.sdk.bbpro.core.peripheral.ConnectionParameters
import com.realsil.sdk.bbpro.core.peripheral.PeripheralParameters
import com.realsil.sdk.bbpro.core.transportlayer.TransportLayer
import com.realsil.sdk.bbpro.vendor.VendorModelCallback
import com.realsil.sdk.core.RtkConfigure
import com.realsil.sdk.core.RtkCore
import com.realsil.sdk.core.logger.ZLogger
import com.realtek.sdk.core.net.wifi.RtkWifiAdapter
import com.realtek.sdk.core.net.wifi.SoftApConfigurationCompat
import com.realtek.sdk.core.net.wifi.SoftApParameter
import com.realtek.sdk.core.net.wifi.WifiConfigurationInfo
import com.realtek.sdk.core.net.wifi.WifiConnectParameter
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Bridges the Realtek Audio Connect SDK to Dart's `GlassDeviceService`.
 *
 * The AI Glass is a Bluetooth **Classic** peripheral speaking Realtek's vendor
 * protocol over SPP — not a BLE GATT device — so the frame button, microphone
 * PCM and photo transfer all arrive on [SmartWearModelCallback] rather than on
 * GATT characteristics. That is why this lives in Kotlin instead of being done
 * with `flutter_blue_plus` in Dart.
 *
 * Section references (§) point at "Realtek Audio Connect SDK Integration Guide
 * -- SmartWear Part" unless marked "WiFi Part".
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

        /** Name fragments used to spot glasses among bonded devices. */
        private const val GLASS_NAME_HINT = "glass"

        /** §5.1.2: in Wi-Fi AP mode the RTSP address is fixed. */
        private const val AP_MODE_RTSP_URL = "rtsp://192.168.43.1:554"

        /** Album under Pictures/ that captures are published to. */
        private const val GALLERY_ALBUM = "BrailleLens"
        private const val MIME_JPEG = "image/jpeg"

        /**
         * How long a frame-button capture stays claimable by Dart. Long enough
         * to cover a slow transfer of a full-size original, short enough that a
         * press abandoned by a screen change does not suppress the next one.
         */
        private const val HARDWARE_CAPTURE_WINDOW_MS = 15_000L

        /** WiFi Part: the reference app's connect timeout. */
        private const val WIFI_CONNECT_TIMEOUT_MS = 30_000L

        /**
         * §5.2.1: SoftAP credentials we ask for in Station mode. Below API 33
         * Android's local-only hotspot picks its own; the real values come back
         * in onSoftApStarted and are what get sent to the glasses.
         */
        private const val SOFT_AP_SSID = "BrailleLens_AP"
        private const val SOFT_AP_PASSWORD = "braille2026"

        @Volatile
        private var sdkInitialised = false

        /**
         * §1 / §2 one-shot SDK bring-up. §1 puts this in the Application
         * class; MainActivity.onCreate calls it for the same "as early as
         * possible" effect, and every bridge entry point re-checks it through
         * [ensureSdk] so ordering can never leave the SDK uninitialised.
         *
         * §1 also lists `autoConnectOnStart` and `functionModuleEnabled`;
         * neither exists on BeeProParams.Builder in AAR 1.8.41, and the
         * v0.5.55 reference app builds exactly the three options used here.
         */
        @Synchronized
        fun initSdk(context: Context): Boolean {
            if (sdkInitialised) return true
            return try {
                val appContext = context.applicationContext
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
                false
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

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

    /**
     * When the frame button last fired the shutter natively, in elapsed millis.
     *
     * The press reaches Dart as BUTTON_CLICKED, and the screens answer it by
     * calling `takePhoto`. Since the native handler has already started the
     * capture, that call rides the in-flight one instead of firing a second
     * shutter — one press stays one photo and one gallery entry. Written from
     * the SDK worker thread, read from the platform thread.
     */
    @Volatile
    private var hardwareCaptureAt: Long = 0L

    /**
     * Whether a frame-button press starts a vendor-channel capture natively.
     *
     * Turned off when Dart answers the press another way — grabbing a frame
     * from the live video feed — so one press does not also queue a
     * full-resolution transfer and put a second image in the gallery.
     * BUTTON_CLICKED is emitted either way.
     */
    @Volatile
    private var hardwareShutterEnabled: Boolean = true

    // Phone-side Wi-Fi (WiFi Part). Created on first live stream only, so a
    // session that never streams never touches the phone's network.
    private var wifiAdapter: RtkWifiAdapter? = null
    private var wifiConnectParameter: WifiConnectParameter? = null
    private var oldWifiConfiguration: WifiConfigurationInfo? = null

    /** Station-mode config parked until the phone SoftAP reports its credentials. */
    private var pendingStationConfig: LiveStreamingConfigInfo? = null

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
        "name" to (bondedName(deviceAddress) ?: "AiSee-G1_01A4"),
    )

    private fun BeeError?.isOk() = this != null && code == 0

    /**
     * Runs [block] on the main thread: inline when already there, posted
     * otherwise. SDK and RtkWifiAdapter callbacks arrive on worker threads
     * (e.g. the SPP reader), but several SDK entry points — initSmartWearDevice
     * first among them — create a Handler internally and crash without a
     * Looper. Running every call back into the SDK on the main thread also
     * keeps the bridge's fields single-threaded with the method channel.
     */
    private fun onMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    // ── SDK lifecycle (§1, §2) ─────────────────────────────────────────────────

    /** [initSdk], reporting a failure to Dart as well as to logcat. */
    private fun ensureSdk(): Boolean =
        initSdk(appContext).also { ok ->
            if (!ok) emitError("Realtek SDK init failed — see logcat", "SDK_INIT")
        }

    // A2DP / HFP proxies, only used to see which bonded device is currently
    // the phone's Bluetooth headset. Bound asynchronously at construction.
    private var a2dpProxy: BluetoothProfile? = null
    private var headsetProxy: BluetoothProfile? = null

    private val profileListener = object : BluetoothProfile.ServiceListener {
        override fun onServiceConnected(profile: Int, proxy: BluetoothProfile) {
            when (profile) {
                BluetoothProfile.A2DP -> a2dpProxy = proxy
                BluetoothProfile.HEADSET -> headsetProxy = proxy
            }
        }

        override fun onServiceDisconnected(profile: Int) {
            when (profile) {
                BluetoothProfile.A2DP -> a2dpProxy = null
                BluetoothProfile.HEADSET -> headsetProxy = null
            }
        }
    }

    init {
        runCatching {
            BluetoothAdapter.getDefaultAdapter()?.let {
                it.getProfileProxy(appContext, profileListener, BluetoothProfile.A2DP)
                it.getProfileProxy(appContext, profileListener, BluetoothProfile.HEADSET)
            }
        }.onFailure { Log.w(TAG, "profile proxies unavailable", it) }
    }

    @SuppressLint("MissingPermission")
    private fun audioConnectedAddresses(): Set<String> = try {
        listOfNotNull(a2dpProxy, headsetProxy)
            .flatMap { it.connectedDevices }
            .mapTo(mutableSetOf()) { it.address }
    } catch (se: SecurityException) {
        emptySet()
    }

    private fun looksLikeGlasses(device: BluetoothDevice): Boolean {
        val n = runCatching { device.name }.getOrNull()?.lowercase().orEmpty()
        return n.contains(GLASS_NAME_HINT) || n.contains("rtk") || n.contains("aiglass")
    }

    /**
     * Every bonded device, most-likely glasses first. The reference app has no
     * name filter — it lets the user pick from an SPP scan — and product names
     * vary by vendor, so nothing is dropped here: devices whose name looks
     * like glasses rank first, then whatever is the phone's active Bluetooth
     * headset (the glasses are one once paired), then the rest.
     */
    @SuppressLint("MissingPermission")
    private fun bondedCandidates(): List<BluetoothDevice> {
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return emptyList()
        if (!adapter.isEnabled) return emptyList()
        return try {
            val audio = audioConnectedAddresses()
            adapter.bondedDevices.orEmpty().sortedWith(
                compareByDescending<BluetoothDevice> { looksLikeGlasses(it) }
                    .thenByDescending { it.address in audio }
            )
        } catch (se: SecurityException) {
            Log.w(TAG, "BLUETOOTH_CONNECT not granted", se)
            emptyList()
        }
    }

    /** Auto-pick for connect-without-address: a name match, else the active headset. */
    private fun defaultGlassesAddress(): String? {
        val candidates = bondedCandidates()
        candidates.firstOrNull { looksLikeGlasses(it) }?.let { return it.address }
        val audio = audioConnectedAddresses()
        return candidates.firstOrNull { it.address in audio }?.address
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

    /**
     * Applied once §2.2's init succeeds.
     *
     * §4.1: the device defaults to 24 kHz mono PCM; the offline STT path wants
     * 16 kHz mono, so ask for that. §3.1.7 marks the device type as required;
     * setting it without calling connectToAI() mirrors the reference app and
     * leaves the (cloud-only) AI session off, which §3.2 asks for when the
     * OpenAI Realtime model is not the one in use.
     */
    private fun applyDeviceParams(client: SmartWearModelClient) {
        runCatching {
            client.setSmartWearDeviceParam(
                SmartWearConfig.Builder()
                    .setDeviceType(SmartWearConstants.DeviceType.TYPE_SINGLE)
                    .setOutputAudioSampleRate(SmartWearConstants.AudioSampleRate.SAMPLE_RATE_16000)
                    .setOutputAudioSampleChannel(SmartWearConstants.AudioChannel.CHANNEL_MONO)
                    .build()
            )
        }.onFailure { Log.w(TAG, "setSmartWearDeviceParam failed", it) }

        runCatching {
            client.setAIModelParam(
                RealtimeApiConfig.Builder()
                    .setSmartDeviceType(RealtimeApiConstants.DeviceType.TYPE_SMART_GLASS)
                    .build()
            )
        }.onFailure { Log.w(TAG, "setAIModelParam failed", it) }
    }

    // ── Method channel ─────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "initialize" -> result.success(ensureSdk())

            "scan" -> {
                if (!ensureSdk()) return result.success(emptyList<Any>())
                val audio = audioConnectedAddresses()
                val found = bondedCandidates().map {
                    hashMapOf<String, Any?>(
                        "address" to it.address,
                        "name" to (runCatching { it.name }.getOrNull() ?: it.address),
                        "likelyGlasses" to looksLikeGlasses(it),
                        "audioConnected" to (it.address in audio),
                    )
                }
                found.forEach { emit("type" to "SCAN_RESULT", *it.toList().toTypedArray()) }
                result.success(found)
            }

            "connect" -> {
                if (!ensureSdk()) return result.success(false)
                val address = call.argument<String>("address")
                    ?: deviceAddress
                    ?: defaultGlassesAddress()
                if (address == null) {
                    emitError(
                        "Could not tell which paired device is the glasses — use Pick device",
                        "NO_DEVICE",
                    )
                    return result.success(false)
                }
                result.success(startConnect(address))
            }

            "disconnect" -> {
                stopLiveStream()
                teardown()
                emitConnectionState("disconnected")
                result.success(null)
            }

            "takePhoto" -> {
                val client = requireClient() ?: return result.success(false)
                // The frame button already fired the shutter natively; that
                // capture is still in flight and its PHOTO_CAPTURED will
                // satisfy the Dart side. Issuing a second startTakePhotoProcess
                // here would put two images in the gallery for one press.
                if (consumeHardwareCapture()) return result.success(true)
                result.success(takePhoto(client, call))
            }

            // Whether the frame button still fires a vendor-channel capture in
            // native code. Dart turns it off when it answers the press with a
            // video-feed snapshot instead.
            "setHardwareShutter" -> {
                hardwareShutterEnabled = call.argument<Boolean>("enabled") ?: true
                if (!hardwareShutterEnabled) hardwareCaptureAt = 0L
                result.success(hardwareShutterEnabled)
            }

            // Publishes an image Dart already holds — a frame grabbed from the
            // live feed, which was never a file — into the same gallery album
            // as the vendor-channel originals.
            "saveToGallery" -> {
                val bytes = call.argument<ByteArray>("data")
                if (bytes == null || bytes.isEmpty()) {
                    return result.error("BAD_ARGS", "data is required", null)
                }
                scope.launch {
                    val uri = withContext(Dispatchers.IO) {
                        saveImageToGallery(appContext, bytes)
                    }
                    result.success(uri)
                }
            }

            "startMic" -> {
                val client = requireClient() ?: return result.success(false)
                // §4.9: no-arg starts the AI-dialogue stream; MEETING is the
                // long-form variant.
                val err = if (call.argument<String>("mode") == "meeting") {
                    client.startUserVoiceInput(SmartWearConstants.UserVoiceMode.MEETING)
                } else {
                    client.startUserVoiceInput()
                }
                result.success(err.isOk())
            }

            "stopMic" -> result.success(modelClient?.stopUserVoiceInput().isOk())

            "startLiveStream" -> result.success(startLiveStream(call))

            "stopLiveStream" -> {
                stopLiveStream()
                result.success(true)
            }

            // §4.18: the result arrives on onReceivedDeviceBatteryInfo.
            "getBattery" -> result.success(modelClient?.getDeviceBatteryInfo().isOk())

            "correctImage" -> correctImage(call, result)

            // Pairing is Android's job; the SDK only connects bonded devices.
            "openBluetoothSettings" -> {
                result.success(
                    runCatching {
                        appContext.startActivity(
                            Intent(Settings.ACTION_BLUETOOTH_SETTINGS)
                                // Started from an application context, so it
                                // needs its own task.
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        )
                    }.onFailure {
                        Log.w(TAG, "could not open Bluetooth settings", it)
                    }.isSuccess
                )
            }

            else -> result.notImplemented()
        }
    }

    private fun requireClient(): SmartWearModelClient? =
        modelClient ?: run {
            emitError("Glasses not connected", "NOT_CONNECTED")
            null
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

        // §2.1: one client per connected BT address.
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
     * SPP is up. §2.2: initialise the SmartWear layer; Dart is told the device
     * is ready only once onSmartWearDeviceInitSuccess lands, because the
     * vendor channel is not usable before that.
     */
    private fun onLinkReady() = onMain {
        // Posted from the SPP worker: the link may have been torn down (user
        // disconnect, or a new connect) before this runs.
        if (connectionManager == null) return@onMain
        try {
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
        } catch (t: Throwable) {
            Log.e(TAG, "initSmartWearDevice failed", t)
            emitError("Glasses init failed: ${t.message}", "DEVICE_INIT")
            emit("type" to "DEVICE_READY", "success" to false)
            emitConnectionState("disconnected")
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
                // Both arrive on the SDK's SPP worker thread.
                PeripheralConnectionManager.STATE_DATA_PREPARED -> onLinkReady()
                PeripheralConnectionManager.STATE_DEVICE_DISCONNECTED -> onMain {
                    releasePhoneWifi(activeLiveChannel)
                    activeLiveChannel = 0
                    emitConnectionState("disconnected")
                }
            }
        }
    }

    // ── Photos (§4.2, §4.3) ────────────────────────────────────────────────────

    /**
     * §4.3 config form of startTakePhotoProcess. Defaults to the full-size
     * original: the thumbnail is far too small for Braille dot detection.
     * Width/height/quality are only sent when Dart passes them, so the device
     * keeps its own defaults otherwise.
     */
    private fun takePhoto(client: SmartWearModelClient, call: MethodCall): Boolean {
        val type = when (call.argument<String>("type")) {
            "thumbnail" -> SmartWearConstants.TakePhotoType.TYPE_TAKE_THUMBNAIL
            "originAndThumbnail" -> SmartWearConstants.TakePhotoType.TYPE_TAKE_ORIGIN_IMAGE_AND_THUMBNAIL
            else -> SmartWearConstants.TakePhotoType.TYPE_TAKE_ORIGIN_IMAGE
        }
        val builder = ThumbnailPhotoConfigInfo.Builder()
            .setTakePhotoType(type)
            // The shutter tone is the wearer's only confirmation the press
            // registered, so it stays on unless Dart says otherwise.
            .setPlayTakePhotoSound(call.argument<Boolean>("playSound") ?: true)
        call.argument<Int>("width")?.let { builder.setPhotoWidth(it) }
        call.argument<Int>("height")?.let { builder.setPhotoHeight(it) }
        call.argument<Int>("quality")?.let { builder.setPhotoQuality(it.coerceIn(0, 9)) }
        return client.startTakePhotoProcess(builder.build()).isOk()
    }

    /**
     * True if the frame button started a capture that Dart has not yet claimed,
     * clearing the marker so only the first caller rides it.
     *
     * The window is generous because the transfer of a full-size original over
     * the vendor channel is slow; a stale marker only ever costs one skipped
     * shutter, while a missing one costs a duplicate image in the dataset.
     */
    private fun consumeHardwareCapture(): Boolean {
        val at = hardwareCaptureAt
        if (at == 0L) return false
        hardwareCaptureAt = 0L
        return android.os.SystemClock.elapsedRealtime() - at < HARDWARE_CAPTURE_WINDOW_MS
    }

    /**
     * Publishes a captured image to the public gallery at
     * `Pictures/BrailleLens` and returns its content URI.
     *
     * Two sources feed this. Vendor-channel originals arrive as a file the SDK
     * wrote into app-private storage, where no gallery app can see them. Video
     * snapshots arrive as bytes straight from the mpv pipeline and were never
     * a file at all. Both end up in the same album so a collection session is
     * one folder on the phone.
     *
     * Runs on a worker thread — it copies the whole image.
     */
    private fun saveImageToGallery(context: Context, imageFile: File): String? {
        if (!imageFile.exists() || imageFile.length() == 0L) return null
        return publishToGallery(context, galleryFileName()) { out ->
            imageFile.inputStream().use { it.copyTo(out) }
        }
    }

    /** Byte-array form, for frames grabbed from the live feed. */
    private fun saveImageToGallery(context: Context, bytes: ByteArray): String? {
        if (bytes.isEmpty()) return null
        return publishToGallery(context, galleryFileName()) { out -> out.write(bytes) }
    }

    /**
     * Timestamped so a collection session sorts chronologically in the gallery
     * and two captures in the same second cannot collide.
     */
    private fun galleryFileName(): String =
        "BrailleLens_${System.currentTimeMillis()}.jpg"

    private fun publishToGallery(
        context: Context,
        displayName: String,
        write: (java.io.OutputStream) -> Unit,
    ): String? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        saveViaMediaStore(context, displayName, write)
    } else {
        saveViaPublicDirectory(context, displayName, write)
    }

    /**
     * API 29+. `RELATIVE_PATH` puts the file in the right album and
     * `IS_PENDING` keeps it hidden until the bytes are all there, so a gallery
     * app scanning mid-copy never shows a half-written JPEG.
     */
    @android.annotation.TargetApi(Build.VERSION_CODES.Q)
    private fun saveViaMediaStore(
        context: Context,
        displayName: String,
        write: (java.io.OutputStream) -> Unit,
    ): String? {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, MIME_JPEG)
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + "/" + GALLERY_ALBUM,
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: return null

        return try {
            resolver.openOutputStream(uri)?.use { write(it) }
                ?: throw java.io.IOException("MediaStore gave no output stream")

            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            uri.toString()
        } catch (t: Throwable) {
            Log.e(TAG, "gallery save failed", t)
            // Leave nothing pending and invisible behind.
            runCatching { resolver.delete(uri, null, null) }
            null
        }
    }

    /**
     * API 24–28, where `RELATIVE_PATH` and `IS_PENDING` do not exist yet.
     * The file is written into the public Pictures directory by hand, then
     * handed to the media scanner so the gallery indexes it — without that
     * scan the image stays invisible until the next reboot.
     *
     * Needs WRITE_EXTERNAL_STORAGE, which the manifest declares with
     * `maxSdkVersion="28"` for exactly this branch.
     */
    private fun saveViaPublicDirectory(
        context: Context,
        displayName: String,
        write: (java.io.OutputStream) -> Unit,
    ): String? = try {
        @Suppress("DEPRECATION")
        val album = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            GALLERY_ALBUM,
        )
        if (!album.exists() && !album.mkdirs()) {
            throw java.io.IOException("could not create ${album.absolutePath}")
        }
        val target = File(album, displayName)
        target.outputStream().use { write(it) }
        MediaScannerConnection.scanFile(
            context,
            arrayOf(target.absolutePath),
            arrayOf(MIME_JPEG),
            null,
        )
        android.net.Uri.fromFile(target).toString()
    } catch (t: Throwable) {
        Log.e(TAG, "legacy gallery save failed", t)
        null
    }

    // ── Image correction (§6) ──────────────────────────────────────────────────

    /**
     * Removes the wide-angle lens distortion. Both image dimensions must be
     * even or the SDK fails the call; the callback runs off the main thread.
     */
    private fun correctImage(call: MethodCall, result: MethodChannel.Result) {
        val input = call.argument<String>("inputPath")
            ?: return result.error("BAD_ARGS", "inputPath is required", null)
        RTKImageToolkit.imageCorrect(
            input,
            call.argument<String>("outputPath"),
            object : ImageCorrectCallback {
                override fun onSuccess(outPath: String) {
                    main.post { result.success(outPath) }
                }

                override fun onFailure(errorCode: Int) {
                    main.post { result.error("IMAGE_CORRECT", "code $errorCode", null) }
                }
            },
        )
    }

    // ── Live streaming (§5) ────────────────────────────────────────────────────

    private fun startLiveStream(call: MethodCall): Boolean {
        val client = requireClient() ?: return false
        val channel = when (call.argument<Int>("channel")) {
            2 -> {
                // The phone hotspot needs location on Android 10-12L, which
                // this app deliberately does not request there (see the
                // manifest). 8-9 have location; 13+ use NEARBY_WIFI_DEVICES.
                if (Build.VERSION.SDK_INT in Build.VERSION_CODES.Q..Build.VERSION_CODES.S_V2) {
                    emitError(
                        "Station mode needs location access on Android 10–12; use Wi-Fi AP mode",
                        "UNSUPPORTED",
                    )
                    return false
                }
                LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_STATION
            }
            3 -> {
                // BT live data is raw H.264 on onReceivedLiveStreamingData and
                // needs the SDK's own player; nothing on the Dart side plays it.
                emitError("Bluetooth live streaming is not supported", "UNSUPPORTED")
                return false
            }
            else -> LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_AP
        }

        // §5.1.1: only the channel and encoding are required; the rest fall
        // back to the device defaults (1280×720, 30 fps, 1 Mbps).
        val config = LiveStreamingConfigInfo().apply {
            liveStreamingChannel = channel
            videoEncoding = LiveStreamingConfigInfo.VideoEncoding.H264
            call.argument<Int>("width")?.let { videoPictureWidth = it }
            call.argument<Int>("height")?.let { videoPictureHeight = it }
            call.argument<Int>("fps")?.let { fps = it }
            call.argument<Int>("bps")?.let { bps = it }
        }
        activeLiveChannel = channel

        return try {
            if (channel == LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_STATION) {
                startPhoneSoftAp(config)
            } else {
                // Result arrives on onStartLiveStreaming.
                client.liveStreamingManager.startLiveStreaming(config)
            }
            true
        } catch (t: Throwable) {
            Log.e(TAG, "live stream failed", t)
            emitError("Live stream failed: ${t.message}", "LIVE_STREAM")
            activeLiveChannel = 0
            false
        }
    }

    /** §5.1.3 / §5.2.4, then hands the phone's Wi-Fi back. */
    private fun stopLiveStream() {
        val channel = activeLiveChannel
        if (channel == 0.toByte()) return
        runCatching { modelClient?.liveStreamingManager?.stopLiveStreaming(channel) }
        releasePhoneWifi(channel)
        activeLiveChannel = 0
    }

    private fun wifi(): RtkWifiAdapter = wifiAdapter ?: RtkWifiAdapter(appContext).also {
        it.registerWiFiAdapterCallback(wifiAdapterCallback)
        wifiAdapter = it
    }

    /**
     * WiFi Part, "Connect Wi-Fi" (§5.1: phone joins the glasses AP through the
     * system API). Parameters are the reference app's defaults. Once joined,
     * RtkWifiAdapter binds the process to that network, which is what lets
     * libmpv's RTSP socket reach 192.168.43.1 even though the AP has no
     * internet.
     */
    private suspend fun joinGlassesAp(ssid: String, password: String): Boolean {
        val adapter = wifi()
        val param = WifiConnectParameter(
            ssid,
            password,
            true,  // appInteractionRequired
            WIFI_CONNECT_TIMEOUT_MS,
            true,  // disableOldNetwork
            true,  // disconnectManually
        )
        wifiConnectParameter = param
        // WiFi Part, "Reconnect Previous Wi-Fi" step 1: remember the network
        // we are about to leave. Only below Android 10, where the join really
        // replaces the phone's network. From 10 on it is an app-scoped
        // WifiNetworkSpecifier request that Android drops on disconnect,
        // handing the phone back to its own network — and reading the old
        // SSID there would need location, which this app does not ask for.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q && oldWifiConfiguration == null) {
            oldWifiConfiguration = runCatching { adapter.getOldWifiConfiguration(param) }.getOrNull()
        }
        return withContext(Dispatchers.IO) {
            try {
                adapter.isSpecificWifiConnected(ssid) || adapter.connect(param)
            } catch (t: Throwable) {
                Log.e(TAG, "Wi-Fi join failed", t)
                false
            }
        }
    }

    /** §5.2.1: the phone hosts the network in Station mode. */
    private fun startPhoneSoftAp(config: LiveStreamingConfigInfo) {
        pendingStationConfig = config
        val adapter = wifi()
        scope.launch(Dispatchers.IO) {
            try {
                adapter.startSoftAp(
                    SoftApParameter.Builder()
                        .ssid(SOFT_AP_SSID)
                        .password(SOFT_AP_PASSWORD)
                        .band(SoftApParameter.BAND_5GHZ)
                        .fallback(true)
                        .traverseChannels(true)
                        .build()
                )
            } catch (t: Throwable) {
                Log.e(TAG, "SoftAP failed", t)
                pendingStationConfig = null
                emitError("Could not start the phone hotspot: ${t.message}", "SOFT_AP")
            }
        }
    }

    private fun releasePhoneWifi(channel: Byte) {
        val adapter = wifiAdapter ?: return
        when (channel) {
            LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_AP -> {
                val old = oldWifiConfiguration
                val param = wifiConnectParameter
                oldWifiConfiguration = null
                // Own scope: dispose() cancels [scope], and the phone must
                // still get its previous network back when the app closes.
                CoroutineScope(Dispatchers.IO).launch {
                    runCatching {
                        // WiFi Part, "Reconnect Previous Wi-Fi" step 2.
                        adapter.disconnect()
                        if (old != null && param != null) {
                            adapter.reconnectOldNetworkCompat(old, param)
                        }
                    }.onFailure { Log.w(TAG, "Wi-Fi restore failed", it) }
                }
            }

            LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_STATION -> {
                pendingStationConfig = null
                runCatching {
                    // §5.2.4 + WiFi Part "disable device sta mode".
                    adapter.stopSoftAp()
                    adapter.updateWifiConnectionState(false)
                    modelClient?.setWifiInfo(
                        WifiApInfo.Builder().mode(WifiApInfo.MODE_IDLE).build()
                    )
                }.onFailure { Log.w(TAG, "SoftAP stop failed", it) }
            }
        }
    }

    private fun emitLiveStream(
        channel: Byte,
        ssid: String?,
        password: String?,
        ipAddress: String?,
        rtspUrl: String,
        wifiConnected: Boolean,
    ) = emit(
        "type" to "LIVE_STREAM",
        "channel" to channel.toInt(),
        "ssid" to ssid,
        "password" to password,
        "ipAddress" to ipAddress,
        "rtspUrl" to rtspUrl,
        "wifiConnected" to wifiConnected,
    )

    private val wifiAdapterCallback = object : RtkWifiAdapter.WiFiAdapterCallback() {
        override fun onWifiConnectionStateChanged(connected: Boolean) {
            super.onWifiConnectionStateChanged(connected)
            emit("type" to "WIFI_CONNECTION", "connected" to connected)
        }

        /** §5.2.2: now that the hotspot is real, send its credentials. */
        override fun onSoftApStarted(softApConfiguration: SoftApConfigurationCompat) {
            super.onSoftApStarted(softApConfiguration)
            // RtkWifiAdapter reports from its own coroutine/receiver thread.
            onMain {
                val config = pendingStationConfig ?: return@onMain
                pendingStationConfig = null
                config.wifiApInfo = WifiApInfo.Builder()
                    .ssid(softApConfiguration.ssid)
                    .password(softApConfiguration.password)
                    .build()
                emit(
                    "type" to "SOFT_AP_STATE",
                    "started" to true,
                    "ssid" to softApConfiguration.ssid,
                    "password" to softApConfiguration.password,
                )
                runCatching { modelClient?.liveStreamingManager?.startLiveStreaming(config) }
                    .onFailure { emitError("Live stream failed: ${it.message}", "LIVE_STREAM") }
            }
        }

        override fun onSoftApFailed(reason: Int) {
            super.onSoftApFailed(reason)
            onMain { pendingStationConfig = null }
            emitError("Phone hotspot failed (reason $reason)", "SOFT_AP")
        }

        override fun onSoftApStopped() {
            super.onSoftApStopped()
            emit("type" to "SOFT_AP_STATE", "started" to false)
        }
    }

    // ── SmartWear callbacks ────────────────────────────────────────────────────

    private val modelCallback = object : SmartWearModelCallback() {

        // SDK worker thread; setAIModelParam and friends may build Handlers.
        override fun onSmartWearDeviceInitSuccess() = onMain {
            modelClient?.let { applyDeviceParams(it) }
            emit("type" to "DEVICE_READY", "success" to true)
            emitConnectionState("connected")
            runCatching { modelClient?.getDeviceBatteryInfo() }
        }

        override fun onSmartWearDeviceInitFail() {
            emit("type" to "DEVICE_READY", "success" to false)
            emitConnectionState("disconnected")
        }

        /**
         * §4.13: the physical frame button. Action type 1 (ACTION_TAKE_PHOTO)
         * is pre-defined, and the SDK routes it here instead of to
         * onReceivedDeviceAction.
         *
         * The press is both surfaced to Dart and acted on natively. Starting
         * the capture here rather than waiting for Dart to answer BUTTON_CLICKED
         * removes a round trip from the wearer's shutter latency, and it means
         * the frame button still produces a gallery image on a screen that is
         * not listening for button presses. [hardwareCaptureAt] keeps the Dart
         * side from firing a second shutter for the same press.
         */
        override fun onDeviceTriggeredTakePhoto() {
            emit("type" to "BUTTON_CLICKED")
            // Dart is handling this press itself (video-feed snapshot).
            if (!hardwareShutterEnabled) return
            val client = modelClient
            if (client == null) {
                emit("type" to "PHOTO_FAILED", "reason" to "glasses not connected")
                return
            }
            hardwareCaptureAt = android.os.SystemClock.elapsedRealtime()
            val started = runCatching {
                client.startTakePhotoProcess(
                    ThumbnailPhotoConfigInfo.Builder()
                        .setTakePhotoType(SmartWearConstants.TakePhotoType.TYPE_TAKE_ORIGIN_IMAGE)
                        .setPlayTakePhotoSound(true)
                        .build()
                ).isOk()
            }.getOrElse {
                Log.e(TAG, "hardware shutter failed", it)
                false
            }
            if (!started) {
                hardwareCaptureAt = 0L
                emit("type" to "PHOTO_FAILED", "reason" to "shutter rejected by device")
            }
        }

        /** §4.13: every other key-press action type. */
        override fun onReceivedDeviceAction(action: Int) {
            emit("type" to "DEVICE_ACTION", "action" to action)
        }

        override fun onStartReceiveUserVoice() =
            emit("type" to "MIC_STATE", "streaming" to true)

        override fun onStopReceiveUserVoice() =
            emit("type" to "MIC_STATE", "streaming" to false)

        /** 16 kHz mono 16-bit PCM, per the SmartWearConfig in [applyDeviceParams]. */
        override fun onReceivedUserVoice(pcmData: ByteArray) =
            emit("type" to "MIC_DATA", "data" to pcmData)

        /**
         * §4.2: some devices return more than one file (e.g. origin plus
         * thumbnail). The largest is the original, which is the one we want.
         */
        override fun onTakePhotoSuccess(photos: MutableList<ThumbnailPhoto>) {
            // The capture this marker was held for has landed, so a later
            // on-screen press gets its own shutter rather than being told to
            // wait for one that is already finished.
            hardwareCaptureAt = 0L
            val file = photos.mapNotNull { it.photoFile }
                .filter { it.exists() }
                .maxByOrNull { it.length() }
            if (file == null) {
                emit("type" to "PHOTO_FAILED", "reason" to "no file returned")
                return
            }
            // Read and copy off the SDK's callback thread: this is a full-size
            // original, and the vendor channel stalls while this thread is busy.
            scope.launch {
                val payload = withContext(Dispatchers.IO) {
                    runCatching {
                        val bytes = file.readBytes()
                        val galleryUri = saveImageToGallery(appContext, file)
                        bytes to galleryUri
                    }
                }
                payload.onSuccess { (bytes, galleryUri) ->
                    if (galleryUri == null) {
                        Log.w(TAG, "photo captured but not published to the gallery")
                    }
                    emit(
                        "type" to "PHOTO_CAPTURED",
                        "data" to bytes,
                        "path" to file.absolutePath,
                        // Empty rather than absent when the gallery write
                        // failed, so Dart can tell "not saved" from "old build".
                        "galleryUri" to (galleryUri ?: ""),
                    )
                }.onFailure {
                    emit("type" to "PHOTO_FAILED", "reason" to (it.message ?: "read failed"))
                }
            }
        }

        override fun onTakePhotoFail(reason: Int) {
            hardwareCaptureAt = 0L
            emit("type" to "PHOTO_FAILED", "reason" to "code $reason")
        }

        /**
         * §4.18: (primary, secondary) battery levels. For a single device such
         * as the AI Glass only the primary value is valid; the SDK reports no
         * charging state.
         */
        override fun onReceivedDeviceBatteryInfo(primaryBattery: Int, secondaryBattery: Int) =
            emit(
                "type" to "BATTERY",
                "level" to primaryBattery,
                "secondaryLevel" to secondaryBattery,
            )

        override fun onWifiApStateChanged(info: WifiApInfo?) = emit(
            "type" to "WIFI_AP_STATE",
            "ssid" to info?.ssid,
            "password" to info?.password,
            "mode" to (info?.mode?.toInt() ?: 0),
        )

        /** WiFi Part, "Synchronize AP information": glasses joined our SoftAP. */
        override fun onWifiStaStateChanged(info: WifiStaInfo?) = emit(
            "type" to "WIFI_STA_STATE",
            "active" to (info?.isActive() ?: false),
            "ipAddress" to info?.ipAddress,
        )

        /**
         * §5.1.1 / §5.2.2 result. Per §5.1.2 the AP-mode RTSP address is fixed
         * and `ipAddress` is only filled in Station mode (§5.2.3), so the URL
         * is built per channel rather than from the IP alone.
         */
        override fun onStartLiveStreaming(
            channel: Byte,
            success: Boolean,
            access: WifiAccessInfo?,
        ) {
            if (!success) {
                emitError("Glasses refused the live stream (channel $channel)", "LIVE_STREAM")
                // SDK worker thread: touch the Wi-Fi adapter and bridge state
                // from main only.
                onMain {
                    releasePhoneWifi(channel)
                    activeLiveChannel = 0
                }
                return
            }

            when (channel) {
                LiveStreamingConfigInfo.LiveStreamingChannel.WIFI_AP -> {
                    val ssid = access?.ssid
                    val password = access?.password.orEmpty()
                    if (ssid.isNullOrEmpty()) {
                        emitError("Glasses did not report their Wi-Fi name", "WIFI_JOIN")
                        emitLiveStream(channel, null, null, null, AP_MODE_RTSP_URL, false)
                        return
                    }
                    // §5.1: join the glasses AP, then connect to RTSP.
                    scope.launch {
                        val joined = joinGlassesAp(ssid, password)
                        if (!joined) {
                            emitError(
                                "Could not join the glasses Wi-Fi \"$ssid\" — join it in Android settings",
                                "WIFI_JOIN",
                            )
                        }
                        emitLiveStream(channel, ssid, password, null, AP_MODE_RTSP_URL, joined)
                    }
                }

                else -> {
                    val ip = access?.ipAddress
                    emitLiveStream(
                        channel,
                        access?.ssid,
                        access?.password,
                        ip,
                        if (ip.isNullOrEmpty()) "" else "rtsp://$ip:554",
                        true,
                    )
                }
            }
        }

        override fun onFileStorageInfoChanged(info: FileStorageInfo?) {
            // Not surfaced to Dart — BrailleLens pulls stills straight over the
            // vendor channel rather than browsing on-glasses storage.
        }
    }

    fun dispose() {
        stopLiveStream()
        teardown()
        runCatching {
            val adapter = BluetoothAdapter.getDefaultAdapter()
            a2dpProxy?.let { adapter?.closeProfileProxy(BluetoothProfile.A2DP, it) }
            headsetProxy?.let { adapter?.closeProfileProxy(BluetoothProfile.HEADSET, it) }
        }
        a2dpProxy = null
        headsetProxy = null
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        events = null
        scope.cancel()
    }
}
