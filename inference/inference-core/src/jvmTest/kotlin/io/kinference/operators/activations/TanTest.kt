package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TanTest {
    private fun getTargetPath(dirName: String) = "tan/$dirName/"

    @Test
    fun test_tan_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tan_example"))
    }

    @Test
    fun test_tan() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tan"))
    }
}
