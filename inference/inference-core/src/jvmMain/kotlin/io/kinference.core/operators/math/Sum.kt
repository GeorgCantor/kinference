package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.sum.sum
import io.kinference.operator.*

sealed class Sum(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SumVer6.VERSION.asRange() -> SumVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Sum operator: $version")
            }
    }
}


class SumVer6(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Sum(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            VariadicIOInfo(0, FLOAT_DATA_TYPES, "data_0", minimumArity = 1)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOAT_DATA_TYPES, "sum", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Sum", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val inputsClean = inputs.filterNotNull().map { it.data as NumberNDArrayCore }
        return listOf(inputsClean.sum().asTensor("sum"))
    }
}


