@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions.activations.elu

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.toPrimitive

@MakePublic
internal suspend fun PrimitiveNDArray.elu(alpha: Float = 1f): PrimitiveNDArray {
    val actualAlpha = alpha.toPrimitive()
    val output = MutablePrimitiveNDArray(strides)

    val inputArray = this.array
    val outputArray = output.array

    val blocksNum = this.array.blocksNum
    val blockSize = this.array.blockSize

    parallelizeByBlocks(blockSize, blocksNum, 2048) { blockStart, blockEnd ->
        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputArray.getBlock(blockIdx) //inputBlocks[blockIdx]
            val outputBlock = outputArray.getBlock(blockIdx) //outputBlocks[blockIdx]

            for (idx in outputBlock.indices) {
                val x = inputBlock[idx]
                if (x >= PrimitiveConstants.ZERO) {
                    outputBlock[idx] = x
                } else {
                    outputBlock[idx] = (FastMath.exp(x) - PrimitiveConstants.ONE) * actualAlpha
                }
            }
        }
    }

    return output
}
