package io.kinference.core.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo

sealed class SequenceEmpty(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KIONNXData<*>, KIONNXSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SequenceEmpty {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SequenceEmptyVer11.VERSION.asRange() -> SequenceEmptyVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SequenceEmpty operator: $version")
            }
        }
    }
}


class SequenceEmptyVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SequenceEmpty(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("dtype", setOf(AttributeProto.AttributeType.INT), required = false, default = TensorProto.DataType.FLOAT.value.toLong()),
        )

        private val INPUTS_INFO = emptyList<IOInfo>()

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("SequenceEmpty", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val dtype by attributeOrNull { it: Number? ->
        TensorProto.DataType.fromValue(it?.toInt() ?: TensorProto.DataType.FLOAT.value)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KIONNXData<*>?>): List<KIONNXSequence> {
        val emptySeq = KIONNXSequence(
            name = "output",
            data = emptyList<KITensor>(),
            info = ValueTypeInfo.SequenceTypeInfo(
                elementType = ValueTypeInfo.TensorTypeInfo(
                    shape = TensorShape.empty(),
                    type = dtype!!
                )
            )
        )
        return listOf(emptySeq)
    }
}
