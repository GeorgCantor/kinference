@file:GeneratePrimitives(
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.bitwise.shift

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.ndarray.stubs.shl
import io.kinference.ndarray.stubs.shr
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.utils.inlines.InlinePrimitive

@MakePublic
internal suspend fun PrimitiveNDArray.bitShift(amountsOfShift: PrimitiveNDArray, direction: BitShiftDirection): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, amountsOfShift.shape))
    return bitShift(amountsOfShift, direction, MutablePrimitiveNDArray(destShape))
}

@MakePublic
internal suspend fun PrimitiveNDArray.bitShift(amountsOfShift: PrimitiveNDArray, direction: BitShiftDirection, destination: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val shiftFunction: PrimitiveType.(Int) -> PrimitiveType
        = when(direction) {
            BitShiftDirection.LEFT -> PrimitiveType::shl
            BitShiftDirection.RIGHT -> PrimitiveType::shr
        }

    return broadcastTwoTensorsPrimitive(this, amountsOfShift, destination) { left: InlinePrimitive, right: InlinePrimitive ->
        InlinePrimitive(left.value.shiftFunction(right.value.toInt()))
    }
}

