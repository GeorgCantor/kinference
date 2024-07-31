package io.kinference.ndarray.extensions.pow

import io.kinference.ndarray.arrays.*
import io.kinference.utils.inlines.InlineInt
import kotlin.math.pow

internal fun Float.pow(x: Double): Float = this.toDouble().pow(x).toFloat()
internal fun Int.pow(x: Double): Int = this.toDouble().pow(x).toInt()
internal fun Long.pow(x: Double): Long = this.toDouble().pow(x).toLong()

private suspend fun FloatNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { _: InlineInt -> pointer.getAndIncrement().toDouble() }
}

private suspend fun IntNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

private suspend fun UIntNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

private suspend fun ByteNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

private suspend fun UByteNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

private suspend fun ShortNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

private suspend fun UShortNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) {_: InlineInt ->  pointer.getAndIncrement().toDouble() }
}

internal suspend fun NumberNDArrayCore.toDoubleNDArray(): DoubleNDArray {
    return when (this) {
        is DoubleNDArray -> this
        is FloatNDArray -> toDoubleNDArray()
        is IntNDArray -> toDoubleNDArray()
        is UIntNDArray -> toDoubleNDArray()
        is ByteNDArray -> toDoubleNDArray()
        is UByteNDArray -> toDoubleNDArray()
        is ShortNDArray -> toDoubleNDArray()
        is UShortNDArray -> toDoubleNDArray()
        else -> error("Unsupported array data type: ${this.type}")
    }
}
