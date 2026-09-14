# Realtek Audio Connect SDK.
# rtk-audioconnect-smartwear ships consumer rules for its own two packages
# (smartwear + ai) but the core/bbpro artifacts are plain jars with none, and the
# SDK reflects over its model and packet classes.
-keep class com.realsil.sdk.** { *; }
-keep class com.realtek.androidx.** { *; }
-dontwarn com.realsil.sdk.**
-dontwarn com.realtek.androidx.**

# The DFU/OTA and demo-support artifacts are deliberately not bundled; silence
# the references the shipped AARs make to them.
-dontwarn com.realsil.sdk.dfu.**
-dontwarn com.realsil.sdk.support.**
