package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.activations.sinh
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class Sinh(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SinhVer9.VERSION.asRange() -> SinhVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Sinh operator: $version")
            }
    }
}


class SinhVer9(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Sinh(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Sinh", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore {
        return when (val type = input.type) {
            DataType.FLOAT -> (input as FloatNDArray).sinh()
            DataType.DOUBLE -> (input as DoubleNDArray).sinh()
            else -> error("Unsupported data type for this operation: $type")
        }
    }
}
