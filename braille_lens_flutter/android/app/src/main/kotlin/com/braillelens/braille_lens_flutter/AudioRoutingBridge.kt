package com.braillelens.braille_lens_flutter

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothProfile
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Forces microphone capture and prompt playback onto the AI Glass headset.
 *
 * The glasses present themselves to Android as an ordinary Bluetooth headset,
 * so this is plain [AudioManager] work and deliberately avoids the Realtek SDK
 * — audio routing keeps working even when the vendor channel is down.
 *
 * The important subtlety is that the mic is only reachable over a **SCO** link,
 * and `startBluetoothSco()` is asynchronous. Returning success immediately
 * would let `speech_to_text` open the built-in mic instead, so
 * [MethodCall] `startSco` does not complete until
 * `ACTION_SCO_AUDIO_STATE_UPDATED` reports CONNECTED (or the caller's timeout
 * elapses).
 */
class AudioRoutingBridge(context: Context, messenger: BinaryMessenger) :
    MethodChannel.MethodCallHandler, EventChannel.StreamHandler {

    companion object {
        private const val TAG = "AudioRoutingBridge"
        private const val CONTROL_CHANNEL = "braille_lens/audio_routing"
        private const val EVENT_CHANNEL = "braille_lens/audio_routing_events"
    }

    private val appContext: Context = context.applicationContext
    private val audio =
        appContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val main = Handler(Looper.getMainLooper())

    private val methodChannel = MethodChannel(messenger, CONTROL_CHANNEL).also {
        it.setMethodCallHandler(this)
    }
    private val eventChannel = EventChannel(messenger, EVENT_CHANNEL).also {
        it.setStreamHandler(this)
    }

    private var events: EventChannel.EventSink? = null
    private var pendingSco: MethodChannel.Result? = null
    private var scoTimeout: Runnable? = null
    private var registered = false

    override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) {
        events = sink
        registerReceiver()
    }

    override fun onCancel(arguments: Any?) {
        events = null
    }

    private fun emit(vararg pairs: Pair<String, Any?>) {
        val map = hashMapOf<String, Any?>(*pairs)
        main.post { events?.success(map) }
    }

    private fun registerReceiver() {
        if (registered) return
        val filter = IntentFilter().apply {
            addAction(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED)
            addAction(BluetoothAdapter.ACTION_CONNECTION_STATE_CHANGED)
        }
        appContext.registerReceiver(receiver, filter)
        registered = true
    }

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED -> {
                    val state = intent.getIntExtra(
                        AudioManager.EXTRA_SCO_AUDIO_STATE,
                        AudioManager.SCO_AUDIO_STATE_ERROR,
                    )
                    val active = state == AudioManager.SCO_AUDIO_STATE_CONNECTED
                    emit("type" to "SCO_STATE", "active" to active)

                    if (active) {
                        // Only now is the headset mic the one a recorder will
                        // actually open.
                        completeSco(true)
                    } else if (state == AudioManager.SCO_AUDIO_STATE_ERROR ||
                        state == AudioManager.SCO_AUDIO_STATE_DISCONNECTED
                    ) {
                        completeSco(false)
                    }
                }

                BluetoothAdapter.ACTION_CONNECTION_STATE_CHANGED -> {
                    val state = intent.getIntExtra(
                        BluetoothAdapter.EXTRA_CONNECTION_STATE,
                        BluetoothAdapter.STATE_DISCONNECTED,
                    )
                    emit(
                        "type" to "HEADSET_STATE",
                        "connected" to (state == BluetoothAdapter.STATE_CONNECTED),
                    )
                }
            }
        }
    }

    private fun completeSco(success: Boolean) {
        scoTimeout?.let { main.removeCallbacks(it) }
        scoTimeout = null
        pendingSco?.let {
            pendingSco = null
            main.post { it.success(success) }
        }
    }

    @SuppressLint("MissingPermission")
    private fun isHeadsetConnected(): Boolean = try {
        val adapter = BluetoothAdapter.getDefaultAdapter()
        adapter != null &&
            adapter.isEnabled &&
            adapter.getProfileConnectionState(BluetoothProfile.HEADSET) ==
            BluetoothProfile.STATE_CONNECTED
    } catch (se: SecurityException) {
        Log.w(TAG, "BLUETOOTH_CONNECT not granted", se)
        false
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "isHeadsetConnected" -> {
                registerReceiver()
                result.success(isHeadsetConnected())
            }

            "startSco" -> {
                registerReceiver()
                if (!isHeadsetConnected()) return result.success(false)

                if (audio.isBluetoothScoOn) return result.success(true)

                // Communication mode is what makes the platform prefer the
                // headset SCO path for both capture and playback.
                audio.mode = AudioManager.MODE_IN_COMMUNICATION
                audio.isBluetoothScoOn = true

                completeSco(false) // clear any earlier pending request
                pendingSco = result

                val timeoutMs = (call.argument<Int>("timeoutMs") ?: 5000).toLong()
                scoTimeout = Runnable {
                    Log.w(TAG, "SCO did not connect within ${timeoutMs}ms")
                    stopSco()
                    completeSco(false)
                }.also { main.postDelayed(it, timeoutMs) }

                try {
                    @Suppress("DEPRECATION")
                    audio.startBluetoothSco()
                } catch (t: Throwable) {
                    Log.e(TAG, "startBluetoothSco failed", t)
                    completeSco(false)
                }
            }

            "stopSco" -> {
                stopSco()
                result.success(null)
            }

            else -> result.notImplemented()
        }
    }

    private fun stopSco() {
        runCatching {
            @Suppress("DEPRECATION")
            audio.stopBluetoothSco()
        }
        runCatching {
            audio.isBluetoothScoOn = false
            audio.mode = AudioManager.MODE_NORMAL
        }
    }

    fun dispose() {
        completeSco(false)
        stopSco()
        if (registered) {
            runCatching { appContext.unregisterReceiver(receiver) }
            registered = false
        }
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        events = null
    }
}
