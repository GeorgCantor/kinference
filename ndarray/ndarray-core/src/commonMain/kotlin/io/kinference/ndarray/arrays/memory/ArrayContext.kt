package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.coroutines.CoroutineContext

class ArrayContext : CoroutineContext.Element {
    val floatStorage = FloatArrayStorage()
    val doubleStorage = DoubleArrayStorage()
    val byteStorage = ByteArrayStorage()
    val shortStorage = ShortArrayStorage()
    val intStorage = IntArrayStorage()
    val longStorage = LongArrayStorage()
    val uByteStorage = UByteArrayStorage()
    val uShortStorage = UShortArrayStorage()
    val uIntStorage = UIntArrayStorage()
    val uLongStorage = ULongArrayStorage()
    val booleanStorage = BooleanArrayStorage()

    companion object Key : CoroutineContext.Key<ArrayContext>

    override val key: CoroutineContext.Key<ArrayContext>
        get() = Key

    fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean = false): MutableNDArrayCore {
        return when(dataType) {
            DataType.FLOAT -> floatStorage.getNDArray(strides, fillZeros)
            DataType.DOUBLE -> doubleStorage.getNDArray(strides, fillZeros)
            DataType.BYTE -> byteStorage.getNDArray(strides, fillZeros)
            DataType.SHORT -> shortStorage.getNDArray(strides, fillZeros)
            DataType.INT -> intStorage.getNDArray(strides, fillZeros)
            DataType.LONG -> longStorage.getNDArray(strides, fillZeros)
            DataType.UBYTE -> uByteStorage.getNDArray(strides, fillZeros)
            DataType.USHORT -> uShortStorage.getNDArray(strides, fillZeros)
            DataType.UINT -> uIntStorage.getNDArray(strides, fillZeros)
            DataType.ULONG -> uLongStorage.getNDArray(strides, fillZeros)
            DataType.BOOLEAN -> booleanStorage.getNDArray(strides, fillZeros)
            else -> error("")
        }
    }

    fun returnNDArray(ndArray: NDArrayCore) {
        when(ndArray.type) {
            DataType.FLOAT -> floatStorage.returnNDArray(ndArray as FloatNDArray)
            DataType.DOUBLE -> doubleStorage.returnNDArray(ndArray as DoubleNDArray)
            DataType.BYTE -> byteStorage.returnNDArray(ndArray as ByteNDArray)
            DataType.SHORT -> shortStorage.returnNDArray(ndArray as ShortNDArray)
            DataType.INT -> intStorage.returnNDArray(ndArray as IntNDArray)
            DataType.LONG -> longStorage.returnNDArray(ndArray as LongNDArray)
            DataType.UBYTE -> uByteStorage.returnNDArray(ndArray as UByteNDArray)
            DataType.USHORT -> uShortStorage.returnNDArray(ndArray as UShortNDArray)
            DataType.UINT -> uIntStorage.returnNDArray(ndArray as UIntNDArray)
            DataType.ULONG -> uLongStorage.returnNDArray(ndArray as ULongNDArray)
            DataType.BOOLEAN -> booleanStorage.returnNDArray(ndArray as BooleanNDArray)
            else -> error("")
        }
    }
}
