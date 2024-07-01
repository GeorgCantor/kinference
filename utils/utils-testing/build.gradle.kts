import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") apply true
}

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("com.squareup.okio:okio:${Versions.okio}")

//                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(kotlin("test-common"))
                api(kotlin("test-annotations-common"))

                api("io.kinference.primitives:primitives-annotations:${Versions.primitives}")
            }
        }

        val jvmMain by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
                api(kotlin("test-junit5"))
            }
        }

        val jsMain by getting {
            dependencies {
                api(kotlin("test-js"))
            }
        }
    }
}
