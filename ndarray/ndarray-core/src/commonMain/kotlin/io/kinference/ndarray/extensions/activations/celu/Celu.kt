package io.kinference.ndarray.extensions.activations.celu

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.parallelizeByBlocks
import kotlin.math.max
import kotlin.math.min

suspend fun FloatNDArray.celu(alpha: Float = 1f): FloatNDArray {
    val output = MutableFloatNDArray(strides)

    val inputArray = this.array
    val blocksNum = this.array.blocksNum
    val blockSize = this.array.blockSize

    val outputArray = output.array

    if (alpha == 1f) {
//        val inputIter = inputBlocks.iterator()
//        val outputIter = outputBlocks.iterator()
        repeat(blocksNum) { blockIdx ->
            val inputBlock = inputArray.getBlock(blockIdx)
            val outputBlock = outputArray.getBlock(blockIdx)

            for (idx in outputBlock.indices) {
                outputBlock[idx] = inputBlock[idx]
            }
        }
    } else {
        parallelizeByBlocks(blockSize, blocksNum, 1048576) { blockStart, blockEnd ->
            for (blockIdx in blockStart until blockEnd) {
                val inputBlock = inputArray.getBlock(blockIdx)
                val outputBlock = outputArray.getBlock(blockIdx)

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = inputBlock[idx] / alpha
                }
            }
        }
    }

    parallelizeByBlocks(blockSize, blocksNum, 2048) { blockStart, blockEnd ->
        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputArray.getBlock(blockIdx)

            for (idx in outputBlock.indices) {
                outputBlock[idx] = FastMath.exp(outputBlock[idx])
            }
        }
    }

    parallelizeByBlocks(blockSize, blocksNum, 1048576) { blockStart, blockEnd ->
        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputArray.getBlock(blockIdx)

            for (idx in outputBlock.indices) {
                outputBlock[idx] = outputBlock[idx] - 1f
            }
        }
    }

    if (alpha != 1f) {
        parallelizeByBlocks(blockSize, blocksNum, 1048576) { blockStart, blockEnd ->
            for (blockIdx in blockStart until blockEnd) {
                val outputBlock = outputArray.getBlock(blockIdx)

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = outputBlock[idx] * alpha
                }
            }
        }
    }

    parallelizeByBlocks(blockSize, blocksNum, 1048576) { blockStart, blockEnd ->
        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputArray.getBlock(blockIdx)
            val outputBlock = outputArray.getBlock(blockIdx)

            for (idx in outputBlock.indices) {
                outputBlock[idx] = max(0f, inputBlock[idx]) + min(0f, outputBlock[idx])
            }
        }
    }

    return output
}
