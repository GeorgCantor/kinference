package io.kinference.ndarray.arrays

typealias StateMarker = (ArrayUsageMarker) -> Unit

const val NO_CONTEXT = "NoContext"

enum class ArrayUsageMarker {
    Unused,
    Used,
    ContextOutput,
    GlobalOutput
}

enum class ArrayTypes(val index: Int) {
    ByteArray(0),
    UByteArray(1),
    ShortArray(2),
    UShortArray(3),
    IntArray(4),
    UIntArray(5),
    LongArray(6),
    ULongArray(7),
    FloatArray(8),
    DoubleArray(9),
    BooleanArray(10);
}

interface MemoryControlledArray {
    fun markContextOutput()
    fun markGlobalOutput()
}
