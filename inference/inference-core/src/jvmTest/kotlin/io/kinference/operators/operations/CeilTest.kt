package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CeilTest {
    private fun getTargetPath(dirName: String) = "ceil/$dirName/"

    @Test
    fun test_ceil() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_ceil"))
    }

    @Test
    fun test_ceil_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_ceil_example"))
    }
}
