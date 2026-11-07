plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.udgamlabs.shruti.sample"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.udgamlabs.shruti.sample"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" } // add "armeabi-v7a" or "x86_64" (emulator) if you need them; they make the app larger
    }

    buildFeatures { buildConfig = true }

    // Keep model files uncompressed in the APK: they barely compress and load faster this way.
    androidResources { noCompress += "onnx" }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // sherpa-onnx 2.0.1, pinned. scripts/get_deps.sh downloads it to app/libs/ and checks its SHA-256.
    implementation(files("libs/sherpa-onnx-2.0.1.aar"))
}
