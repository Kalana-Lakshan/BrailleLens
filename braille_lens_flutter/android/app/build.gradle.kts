plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.braillelens.braille_lens_flutter"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // kotlinOptions {
    //     jvmTarget = JavaVersion.VERSION_17.toString()
    // }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.braillelens.braille_lens_flutter"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    packaging {
        resources {
            // The Realtek AARs each ship their own copies of these.
            excludes += setOf(
                "META-INF/DEPENDENCIES",
                "META-INF/LICENSE*",
                "META-INF/NOTICE*",
                "META-INF/*.kotlin_module",
            )
        }
        jniLibs {
            // rtk-audioconnect-smartwear bundles the RTSP/H.264 native decoder.
            useLegacyPackaging = true
        }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // Realtek Audio Connect SDK — the AI Glass vendor channel, live streaming
    // and DFU. Copied from the AIGlass reference app (v0.5.55) into app/libs.
    implementation(files("libs/rtk-androidx-core-1.0.26.jar"))
    implementation(files("libs/rtk-audioconnect-common-1.15.28.aar"))
    implementation(files("libs/rtk-audioconnect-core-1.9.10.jar"))
    implementation(files("libs/rtk-audioconnect-smartwear-1.8.41.aar"))
    implementation(files("libs/rtk-core-ktx-1.7.83.jar"))
    implementation(files("libs/rtk-dfu-3.14.36.jar"))
    implementation(files("libs/rtk-support-1.7.94.aar"))
    // Local AAR libraries from android/app/libs/
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))

    // SmartWear guide §1 step 2: third-party libraries the SDK needs.
    implementation("com.squareup.okhttp3:okhttp:4.9.3")
    implementation("com.google.code.gson:gson:2.10.1")

    // Transitive requirements of the SDK AARs (they are plain file deps, so
    // nothing is resolved for them automatically).
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.preference:preference-ktx:1.2.1")
    implementation("androidx.work:work-runtime-ktx:2.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("com.google.android.material:material:1.11.0")
    implementation("com.google.guava:guava:31.1-android")
}

flutter {
    source = "../.."
}
