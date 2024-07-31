package io.kinference.core.model

import io.kinference.core.KIONNXData
import io.kinference.core.graph.KIGraph
import io.kinference.core.markOutput
import io.kinference.graph.Contexts
import io.kinference.model.Model
import io.kinference.ndarray.arrays.memory.*
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.*
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.*

class KIModel(
    val id: String,
    val name: String,
    val opSet: OperatorSetRegistry,
    val graph: KIGraph,
    memoryLimiter: MemoryLimiter = MemoryLimiters.NoAllocator,
    parallelismLimit: Int = PlatformUtils.cores,
) : Model<KIONNXData<*>>, Profilable, Cacheable {
    private val profiles: MutableList<ProfilingContext> = ArrayList()

    @OptIn(ExperimentalCoroutinesApi::class)
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(parallelismLimit)
    private val modelArrayStorage: ModelArrayStorage = ModelArrayStorage(memoryLimiter)

    override fun addProfilingContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override suspend fun predict(input: List<KIONNXData<*>>, profile: Boolean): Map<String, KIONNXData<*>> {
        val contexts = Contexts<KIONNXData<*>>(
            null,
            if (profile) addProfilingContext("Model $name") else null
        )

        val limiterContext = ParallelismLimiterContext(dispatcher)
        var coreReserved = false
        val results = try {
            withContext(NonCancellable) {
                ResourcesDispatcher.reserveCore()
                coreReserved = true
            }

            val allocatorContext = modelArrayStorage.createAllocatorContext()
            val mixedContext = allocatorContext + limiterContext

            withContext(mixedContext) {
                val coroutineContext = coroutineContext[AllocatorContext.Key]!!
                val execResult = graph.execute(input, contexts)
                execResult.forEach { it.markOutput() }
                coroutineContext.closeAllocated()
                execResult
            }
        } finally {
            if (coreReserved) {
                ResourcesDispatcher.releaseCore()
            }
        }

        return results.associateBy { it.name!! }
    }

    override suspend fun close() {
        graph.close()
        modelArrayStorage.close()
    }

    override fun clearCache() {
        modelArrayStorage.clearCache()
    }

    companion object {
        private val modelCounter = atomic(0)

        private fun generateModelId(): Int = modelCounter.incrementAndGet()

        suspend operator fun invoke(
            proto: ModelProto,
            memoryLimiter: MemoryLimiter = MemoryLimiters.NoAllocator,
            limiterParallelismCounter: Int = PlatformUtils.cores,
        ): KIModel {
            val name = "${proto.domain}:${proto.modelVersion}"
            val id = "$name:${generateModelId()}"
            val opSet = OperatorSetRegistry(proto.opSetImport)
            val graph = KIGraph(proto.graph!!, opSet)
            return KIModel(id, name, opSet, graph, memoryLimiter, limiterParallelismCounter)
        }
    }
}
