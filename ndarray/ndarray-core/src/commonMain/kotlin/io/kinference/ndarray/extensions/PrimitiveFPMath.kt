@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import kotlin.math.*

@MakePublic
internal suspend fun PrimitiveNDArray.isNaN(): BooleanNDArray = predicateElementWise { it.isNaN() }

@MakePublic
internal suspend fun PrimitiveNDArray.ceil(): PrimitiveNDArray = applyElementWise { ceil(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.floor(): PrimitiveNDArray = applyElementWise { floor(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.round(): PrimitiveNDArray = applyElementWise { round(it) }

@MakePublic
internal suspend fun PrimitiveNDArray.sqrt(): PrimitiveNDArray = applyElementWise { sqrt(it) }

@MakePublic
internal suspend fun PrimitiveNDArray.reciprocal(): PrimitiveNDArray = applyElementWise { PrimitiveConstants.ONE / it }
