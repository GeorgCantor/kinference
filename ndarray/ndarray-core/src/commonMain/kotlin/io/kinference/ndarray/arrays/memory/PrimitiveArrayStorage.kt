@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray

@GenerateNameFromPrimitives
@MakePublic
internal class PrimitiveArrayStorage {
    private val storage = HashMap<Int, ArrayDeque<PrimitiveArray>>(2)

    fun getNDArray(strides: Strides, fillZeros: Boolean = false): MutablePrimitiveNDArray {
        val blockSize = blockSizeByStrides(strides)
        val blocksNum = strides.linearSize / blockSize

        val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        val blocks = Array(blocksNum) {
            val block = queue.removeFirstOrNull()
            if (fillZeros) {
                block?.fill(PrimitiveConstants.ZERO)
            }
            block ?: PrimitiveArray(blockSize)
        }

        val tiled = PrimitiveTiledArray(blocks)

        return MutablePrimitiveNDArray(tiled, strides)
    }

    fun returnNDArray(ndarray: PrimitiveNDArray) {
        val blockSize = ndarray.array.blockSize
        val blocksNum = ndarray.array.blocksNum

        val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        queue.addAll(ndarray.array.blocks)
    }
}
