package org.nd4j.linalg.factory

import java.lang.reflect.Field
import java.util.concurrent.ConcurrentHashMap

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.shape.LongShapeDescriptor
import org.nd4j.linalg.cpu.nativecpu.DirectShapeInfoProvider
import org.nd4j.linalg.primitives.Pair

/**
  * @see https://github.com/deeplearning4j/deeplearning4j/issues/7125
  */
object CpuBackendNd4jPurger {

  def purge(): Unit = {
    val shapeInfoProvider = Nd4j.getShapeInfoProvider.asInstanceOf[DirectShapeInfoProvider]
    val field: Field = shapeInfoProvider.getClass.getDeclaredField("longCache")
    field.setAccessible(true)
    val longCache = new ConcurrentHashMap[LongShapeDescriptor, Pair[DataBuffer, Array[Long]]]()
    field.set(shapeInfoProvider, longCache)
    shapeInfoProvider.purgeCache()
    Nd4j.getExecutioner.getTADManager.purgeBuffers()
  }

}
