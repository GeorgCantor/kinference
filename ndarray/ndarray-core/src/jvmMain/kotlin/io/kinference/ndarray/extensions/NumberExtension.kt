package io.kinference.ndarray.extensions

import kotlin.math.sqrt

import kotlin.math.ln

internal val Byte.Companion.MAX_VALUE_FOR_MIN: Byte
    get() = MAX_VALUE

internal val Short.Companion.MAX_VALUE_FOR_MIN: Short
    get() = MAX_VALUE

internal val Int.Companion.MAX_VALUE_FOR_MIN: Int
    get() = MAX_VALUE

internal val Long.Companion.MAX_VALUE_FOR_MIN: Long
    get() = MAX_VALUE

internal val UByte.Companion.MAX_VALUE_FOR_MIN: UByte
    get() = MAX_VALUE

internal val UShort.Companion.MAX_VALUE_FOR_MIN: UShort
    get() = MAX_VALUE

internal val UInt.Companion.MAX_VALUE_FOR_MIN: UInt
    get() = MAX_VALUE

internal val ULong.Companion.MAX_VALUE_FOR_MIN: ULong
    get() = MAX_VALUE

internal val Float.Companion.MAX_VALUE_FOR_MIN: Float
    get() = POSITIVE_INFINITY

internal val Double.Companion.MAX_VALUE_FOR_MIN: Double
    get() = POSITIVE_INFINITY

internal val Byte.Companion.MIN_VALUE_FOR_MAX: Byte
    get() = MIN_VALUE

internal val Short.Companion.MIN_VALUE_FOR_MAX: Short
    get() = MIN_VALUE

internal val Int.Companion.MIN_VALUE_FOR_MAX: Int
    get() = MIN_VALUE

internal val Long.Companion.MIN_VALUE_FOR_MAX: Long
    get() = MIN_VALUE

internal val UByte.Companion.MIN_VALUE_FOR_MAX: UByte
    get() = MIN_VALUE

internal val UShort.Companion.MIN_VALUE_FOR_MAX: UShort
    get() = MIN_VALUE

internal val UInt.Companion.MIN_VALUE_FOR_MAX: UInt
    get() = MIN_VALUE

internal val ULong.Companion.MIN_VALUE_FOR_MAX: ULong
    get() = MIN_VALUE

internal val Float.Companion.MIN_VALUE_FOR_MAX: Float
    get() = NEGATIVE_INFINITY

internal val Double.Companion.MIN_VALUE_FOR_MAX: Double
    get() = NEGATIVE_INFINITY

internal inline fun abs(x: UInt) = x

internal inline fun abs(x: ULong) = x

internal inline fun sqrt(x: Int) = sqrt(x.toDouble()).toInt()

internal inline fun sqrt(x: Long) = sqrt(x.toDouble()).toLong()

internal inline fun sqrt(x: UInt) = sqrt(x.toDouble()).toUInt()

internal inline fun sqrt(x: ULong) = sqrt(x.toDouble()).toULong()

internal inline fun ln(x: Int) = ln(x.toDouble()).toInt()

internal inline fun ln(x: Long) = ln(x.toDouble()).toLong()

internal inline fun ln(x: UInt) = ln(x.toDouble()).toUInt()

internal inline fun ln(x: ULong) = ln(x.toDouble()).toULong()
