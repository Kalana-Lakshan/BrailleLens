package com.braillelens.braille_lens_flutter

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.util.Log
import com.realsil.sdk.audioconnect.smartwear.SmartWearModelProxy
import com.realsil.sdk.bbpro.BeeProParams
import com.realsil.sdk.bbpro.MultiPeripheralConnectionManager
import com.realsil.sdk.bbpro.core.transportlayer.TransportLayer
import com.realsil.sdk.core.RtkConfigure
import com.realsil.sdk.core.RtkCore
import com.realsil.sdk.core.logger.ZLogger

/**
 * Application entry point, responsible for bringing up the Realtek Audio Connect stack.
 *
 * The SDK is initialised here rather than in [MainActivity] because
 * `MultiPeripheralConnectionManager` and `SmartWearModelProxy` are process-wide
 * singletons that must exist before any Activity binds to a device, and because
 * initialising from an Activity re-runs on every recreation.
 *
 * Ordering matters: `RtkCore.initialize` must run first — every other Realtek
 * module reads its configuration and logger from it.
 */
class MainApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Plugins such as camera/audio can spawn auxiliary processes; the Bluetooth
        // stack must only be initialised in the main one.
        if (!isMainProcess()) {
            Log.i(TAG, "skipping Realtek init in secondary process")
            return
        }

        // Derived from the manifest rather than BuildConfig, which AGP 8 no longer
        // generates unless buildFeatures.buildConfig is turned back on.
        val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0

        try {
            // 1. rtk-core. Mandatory, and must precede every other Realtek module.
            val configure = RtkConfigure.Builder()
                .debugEnabled(debuggable)
                .printLog(debuggable)
                .globalLogLevel(if (debuggable) ZLogger.DEBUG else ZLogger.WARN)
                .logTag(TAG)
                .build()
            RtkCore.initialize(this, configure)

            // 2. Audio Connect. A2DP + HFP are what put Flutter TTS on the glasses'
            //    speakers: once the profiles are connected the glasses become the
            //    active audio route and the normal media/voice stream follows.
            //
            //    NOTE: the integration notes listed .autoConnectOnStart(false) and
            //    .functionModuleEnabled(true); neither exists on BeeProParams.Builder
            //    in rtk-audioconnect-common 1.15.28, so they are omitted. The
            //    available options are serverEnabled / listenA2dp / listenHfp /
            //    bindHfpDisconnection / connectA2dp / syncDataWhenConnected /
            //    uuid / transport.
            val params = BeeProParams.Builder()
                .syncDataWhenConnected(true)
                .connectA2dp(true)
                .listenHfp(true)
                .build()
            MultiPeripheralConnectionManager.getInstance(this).initialize(params)

            // 3. SmartWear model registry — hands out a SmartWearModelClient per device.
            SmartWearModelProxy.initialize(this)

            // Transport-layer tracing is very chatty and the vendor explicitly warns it
            // degrades image transmission and real-time voice streaming. Keep it off.
            TransportLayer.TDBG = false

            Log.i(TAG, "Realtek Audio Connect SDK initialised")
        } catch (t: Throwable) {
            // BrailleLens is usable without the glasses (phone camera + phone mic),
            // so a missing or broken SDK must not prevent the app from starting.
            Log.e(TAG, "Realtek SDK initialisation failed", t)
        }
    }

    private fun isMainProcess(): Boolean {
        val current = currentProcessName() ?: return true
        return current == packageName
    }

    private fun currentProcessName(): String? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) return getProcessName()
        val pid = android.os.Process.myPid()
        val am = getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return null
        return am.runningAppProcesses?.firstOrNull { it.pid == pid }?.processName
    }

    companion object {
        private const val TAG = "BrailleLensGlasses"
    }
}
