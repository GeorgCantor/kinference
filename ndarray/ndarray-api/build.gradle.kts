import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        commonMain {
            dependencies {
                api(project(":utils:utils-common"))
                api(libs.kinference.primitives.annotations)
            }
        }
    }
}
