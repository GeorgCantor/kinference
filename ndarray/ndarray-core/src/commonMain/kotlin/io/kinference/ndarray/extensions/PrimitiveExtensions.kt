@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.acceptWithRecursive
import io.kinference.ndarray.stubs.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.math.*

@MakePublic
internal fun erf(value: PrimitiveType): PrimitiveType {
    val sign = value.toDouble().sign
    val doubleValue = abs(value.toDouble())
    val t = 1 / (1 + ERF_P_VALUE * doubleValue)

    val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

    return (sign * (1.0 - sum * exp(-doubleValue * doubleValue))).toPrimitive()
}


@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE])
@BindPrimitives(type1 = [DataType.BYTE, DataType.UBYTE])
@MakePublic
internal suspend fun PrimitiveNDArray.quantizeDot(other: @BindPrimitives.Type1 PrimitiveNDArray, destination: MutableFloatNDArray, zeroPointA: Int = 0, zeroPointB: Int = 0, scale: Float = 1f, coroutineContext: CoroutineContext = EmptyCoroutineContext): MutableFloatNDArray {
    val M = this.shape[0]

    suspend fun wrapper(body: suspend (inner: suspend () -> Unit) -> Unit = { it() }) {
        for (rdBlockNum in 0 until destination.blocksInRow) {
            body {
                for (i in 0 until M) {
                    val dBlockOffset = i * destination.blocksInRow
                    val lBlockOffset = i * this.blocksInRow

                    var k = 0
                    for (lBlockNum in 0 until this.blocksInRow) {
                        val lBlock = this.array.getBlock(lBlockOffset + lBlockNum)
                        for (lInd in lBlock.indices) {
                            val temp = lBlock[lInd].toInt() - zeroPointA
                            val rBlockOffset = k * other.blocksInRow
                            val rBlock = other.array.getBlock(rBlockOffset + rdBlockNum)
                            val dBlock = destination.array.getBlock(dBlockOffset + rdBlockNum)
                            for (idx in rBlock.indices) {
                                dBlock[idx] += (temp * (rBlock[idx].toInt() - zeroPointB)) * scale
                            }
                            k++
                        }
                    }
                }
            }
        }
    }

    if (other.blocksInRow > 1) {
        coroutineScope { wrapper { launch { it() } } }
    } else {
        wrapper()
    }

    return destination
}


@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE, DataType.INT])
@MakePublic
internal fun PrimitiveNDArray.withZeroPoint(zeroPoint: PrimitiveNDArray): IntNDArray {
    return if (zeroPoint.linearSize == 1) {
        val zero = zeroPoint.singleValue().toInt()
        val arr = IntTiledArray(this.strides)
        arr.pointer().accept(array.pointer(), arr.size) { _, src -> src.toInt() - zero }
        IntNDArray(arr, strides)
    } else {
        val arr = IntTiledArray(strides)
        arr.pointer().acceptWithRecursive(this.array.pointer(), zeroPoint.array.pointer(), arr.size) { _, src, zero -> src.toInt() - zero.toInt() }
        IntNDArray(arr, strides)
    }
}

@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE])
@MakePublic
internal fun PrimitiveNDArray.dequantize(zeroPoint: PrimitiveNDArray?, scale: FloatNDArray, axis: Int?): FloatNDArray {
    val zeros = zeroPoint?.array
    val output = MutableFloatNDArray(FloatTiledArray(this.array.size, this.array.blockSize), this.strides)

    when {
        canDequantizePerTensor(zeroPoint, scale) -> {
            val zero = if (zeros == null) 0f else zeros[0].toFloat()
            val sc = scale.singleValue()

            output.array.pointer().accept(this.array.pointer(), output.linearSize) { _, src -> (src.toFloat() - zero) * sc }
        }
        canDequantizePerAxis(axis!!, zeroPoint, scale) -> {
            val actualAxis = indexAxis(axis)
            val blockCount = computeBlockSize(toDim = actualAxis)
            val blockSize = computeBlockSize(fromDim = actualAxis + 1)
            var outOffset = 0
            repeat(blockCount) {
                val zeroPointer = zeros?.pointer()
                val scalePointer = scale.array.pointer()
                for (i in 0 until shape[actualAxis]) {
                    val zero = zeroPointer?.getAndIncrement()?.toFloat() ?: 0f
                    val sc = scalePointer.getAndIncrement()

                    output.array.pointer(outOffset).accept(this.array.pointer(outOffset), blockSize) { _, src -> (src.toFloat() - zero) * sc }
                    outOffset += blockSize
                }
            }
        }
        else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors containing ${shape[axis]} elements")
    }

    return output
}

@SpecifyPrimitives(include = [DataType.FLOAT, DataType.DOUBLE])
@MakePublic
internal suspend fun PrimitiveNDArray.dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
    other as PrimitiveNDArray; destination as MutablePrimitiveNDArray

    val alpha = alpha.toPrimitive()
    val dBlocksInRow = destination.blocksInRow
    val lrBlocksInRow = this.blocksInRow

    val n = this.shape[0]
    val t = this.shape[1]
    val m = other.shape[0]

    val dBlockSize = destination.array.blockSize
    val lrBlockSize = this.array.blockSize

    val destArray = destination.array
    val leftArray = this.array
    val rightArray = other.array
    val rowFlop = t * m
    val zero = (0).toPrimitive()

    // Constant 262144 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(rowFlop, n, 262144) { nStart: Int, nEnd: Int ->
        val mSums = Array(m) { PrimitiveArray(lrBlockSize) }
        for (i in nStart until nEnd) {
            val leftBlockOffset = i * lrBlocksInRow
//            val rightBlockIter = rightArray.iterator()

            val destBlockOffset = i * dBlocksInRow

            for (k in 0 until m) {
                val tempArray = mSums[k]
                val rightBlockOffset = k * lrBlocksInRow
                for (lrBlock in 0 until lrBlocksInRow) {
                    val leftBlock = leftArray.getBlock(leftBlockOffset + lrBlock)
                    val rightBlock = rightArray.getBlock(rightBlockOffset + lrBlock)

                    for (j in tempArray.indices) {
                        tempArray[j] += leftBlock[j] * rightBlock[j]
                    }
                }
            }

            val mSumsIter = mSums.iterator()
            for (destBlockNum in 0 until dBlocksInRow) {
                val destBlock = destArray.getBlock(destBlockOffset + destBlockNum)
                for (j in destBlock.indices) {
                    val sumBlock = mSumsIter.next()
                    destBlock[j] = sumBlock.sum() * alpha
                    sumBlock.fill(zero)
                }
            }
        }
    }

    return destination
}
