package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.layer.recurrent.LayerDirection
import io.kinference.core.optimizer.rules.context.LSTMContextRule
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.operator.*
import io.kinference.optimizer.GraphOptimizer.Companion.isOpt
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class LSTM(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): LSTM {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in LSTMVer7.VERSION.asRange() -> LSTMVer7(name, attributes, inputs, outputs)
                else -> error("Unsupported version of LSTM operator: $version")
            }
        }
    }
}

class LSTMVer7(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : LSTM(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("activation_alpha", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activation_beta", setOf(AttributeProto.AttributeType.FLOATS), false, emptyList<Float>()),
            AttributeInfo("activations", setOf(AttributeProto.AttributeType.STRINGS), false, listOf("Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh")),
            AttributeInfo("clip", setOf(AttributeProto.AttributeType.FLOAT), false, Float.MAX_VALUE),
            AttributeInfo("direction", setOf(AttributeProto.AttributeType.STRING), false, "forward"),
            AttributeInfo("hidden_size", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("input_forget", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("layout", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false), // [seq_length, batch_size, input_size]
            IOInfo(1, TYPE_CONSTRAINTS, "W", optional = false), // [num_directions, 4*hidden_size, input_size]
            IOInfo(2, TYPE_CONSTRAINTS, "R", optional = false), // [num_directions, 4*hidden_size, hidden_size]
            IOInfo(3, TYPE_CONSTRAINTS, "B", optional = true), // [num_directions, 8*hidden_size]

            IOInfo(4, setOf(TensorProto.DataType.INT32), "sequence_lens", optional = true), // [batch_size]
            IOInfo(5, TYPE_CONSTRAINTS, "initial_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(6, TYPE_CONSTRAINTS, "initial_c", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(7, TYPE_CONSTRAINTS, "P", optional = true) // [num_directions, 3*hidden_size]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = true), // [seq_length, num_directions, batch_size, hidden_size]
            IOInfo(1, TYPE_CONSTRAINTS, "Y_h", optional = true), // [num_directions, batch_size, hidden_size]
            IOInfo(2, TYPE_CONSTRAINTS, "Y_c", optional = true) // [num_directions, batch_size, hidden_size]
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("LSTM", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val activations: List<String> by attribute { it: List<String> ->
        when(direction) {
            LayerDirection.FORWARD, LayerDirection.REVERSE -> it.subList(0, 3)
            LayerDirection.BIDIRECTIONAL -> it
        }
    }

    private val direction: LayerDirection by attribute { it: String -> LayerDirection.valueOf(it.uppercase()) }

    private val hiddenSize: Int by attribute("hidden_size") { it: Number -> it.toInt() }
    private val batchWise: Boolean by attribute("layout") { it: Number -> it.toInt() == 1 }

    init {
        if (batchWise) error("BatchWise LSTM not supported")
    }

    private val lstmLayer = LSTMLayerBase.create(hiddenSize, activations, direction)

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!
        val inputAsLSTMInput = DefaultLSTMInput(input.data as NumberNDArrayCore)

        val weights = inputs[1]!!
        val preparedWeights = weights.takeIf { isOpt(it.name) } ?: LSTMContextRule.prepareWeights(weights)
        val weightsAsLSTMWeights = DefaultLSTMWeights(preparedWeights.data as NumberNDArrayCore)

        val recurrentWeights = inputs[2]!!
        val preparedRecurrentWeights = recurrentWeights.takeIf { isOpt(it.name) }
            ?: LSTMContextRule.prepareWeights(recurrentWeights)
        val recurrentWeightsAsLSTMWeights = DefaultLSTMWeights(preparedRecurrentWeights.data as NumberNDArrayCore)

        val bias = inputs.getOrNull(3)
        val preparedBias = bias?.let { tensor ->
            tensor.takeIf { isOpt(it.name) } ?: LSTMContextRule.prepareBias(tensor)
        }

        val peepholes = inputs.getOrNull(7)
        val preparedPeepholes = peepholes?.let { tensor ->
            tensor.takeIf { isOpt(it.name) } ?: LSTMContextRule.preparePeepholes(tensor)
        }

        val sequenceLens = inputs.getOrNull(4)
        val initialState = inputs.getOrNull(5)
        val initialCellState = inputs.getOrNull(6)

        val (output, lastState, lastCellState) = lstmLayer.apply(
            inputAsLSTMInput,
            weightsAsLSTMWeights,
            recurrentWeightsAsLSTMWeights,
            preparedBias?.data as NumberNDArrayCore?,
            sequenceLens?.data as IntNDArray?,
            initialState?.data as NumberNDArrayCore?,
            initialCellState?.data as NumberNDArrayCore?,
            preparedPeepholes?.data as NumberNDArrayCore?,
            input.data.type
        )
        
        return listOf(output.asTensor("Y"), lastState.asTensor("Y_h"), lastCellState.asTensor("Y_c"))
    }
}
