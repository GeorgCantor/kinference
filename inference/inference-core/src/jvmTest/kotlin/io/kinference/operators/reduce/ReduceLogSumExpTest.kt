package io.kinference.operators.reduce

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ReduceLogSumExpVer18Test {
    private fun getTargetPath(dirName: String) = "reduce_log_sum_exp/v18/$dirName/"

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_example_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_random_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_example_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_example_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_random_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_random_expanded()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_random_expanded"))
    }
}

class ReduceLogSumExpVer1Test {
    private fun getTargetPath(dirName: String) = "reduce_log_sum_exp/v1/$dirName/"

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_default_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_default_axes_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_do_not_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_do_not_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_keepdims_random"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_example()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_example"))
    }

    @Test
    fun test_reduce_log_sum_exp_negative_axes_keepdims_random()  = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_reduce_log_sum_exp_negative_axes_keepdims_random"))
    }
}
