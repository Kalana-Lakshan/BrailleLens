package com.braillelens.braille_lens_flutter

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {

    private var glassBridge: GlassBridge? = null
    private var audioRoutingBridge: AudioRoutingBridge? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        glassBridge = GlassBridge(applicationContext, messenger)
        audioRoutingBridge = AudioRoutingBridge(applicationContext, messenger)
    }

    override fun onDestroy() {
        // Release the SPP link and drop the SCO route; leaving either up keeps
        // the glasses mic hot after the app is gone.
        glassBridge?.dispose()
        glassBridge = null
        audioRoutingBridge?.dispose()
        audioRoutingBridge = null
        super.onDestroy()
    }
}
