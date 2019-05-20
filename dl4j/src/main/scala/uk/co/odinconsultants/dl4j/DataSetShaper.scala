package uk.co.odinconsultants.dl4j4s.data

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.reflect.ClassTag

object DataSetShaper {
  type OneHotMatrix2Cat = (Seq[(Int, Int)], Int)
}

class DataSetShaper[T : Numeric : ClassTag] {

  import DataSetShaper._

  val opT: Numeric[T] = implicitly[Numeric[T]]

  type Series2Cat = (Seq[T], Int)

  /**
    * Aha! Was the victim of this bug: https://github.com/deeplearning4j/dl4j-examples/issues/779
    */
  def to3DDataset(s2cs: Seq[Series2Cat], nClasses: Int, seriesLength: Int, nIn: Int): DataSet = {
    val n         = s2cs.size
    val features  = Nd4j.zeros(n, nIn, seriesLength)
    val labels    = Nd4j.zeros(n, nClasses, seriesLength)

    s2cs.zipWithIndex.foreach { case ((xs, c), i) =>
      xs.zipWithIndex.foreach { case (x, j) =>
        val indxFeatures: Array[Int] = Array(i, 0, j)
        features.putScalar(indxFeatures, opT.toDouble(x))
        val indxLabels:   Array[Int] = Array(i, c, j)
        labels.putScalar(indxLabels, 1)
      }
    }
    new DataSet(features, labels)
  }

  def to4DDataset(s2cs: Seq[OneHotMatrix2Cat], nClasses: Int, w: Int, h: Int): DataSet = {
    val n         = s2cs.size
    val nChannels = 1
    val features  = Nd4j.zeros(n, nChannels, w, h)
    val labels    = Nd4j.zeros(n: Long, nClasses: Long)

    s2cs.zipWithIndex.foreach { case ((coords, c), i) =>
      coords.zipWithIndex.foreach { case ((x, y), j) =>
        val indxFeatures: Array[Int] = Array(i, 0, x, y)
        features.putScalar(indxFeatures, 1d)
        val indxLabels:   Array[Int] = Array(i, c)
        labels.putScalar(indxLabels, 1)
      }
    }
    new DataSet(features, labels)
  }

  def to2DDataset(s2cs: Seq[Series2Cat], nClasses: Int, nFeatures: Int): DataSet = {
    val nSamples  = s2cs.size
    val features  = Nd4j.zeros(Array[Int](nSamples, nFeatures): _*)
    val labels    = Nd4j.zeros(Array[Int](nSamples, nClasses): _*)

    s2cs.zipWithIndex.foreach { case ((xs, c), i) =>
      xs.zipWithIndex.foreach { case (x, j) =>
        val indxFeatures: Array[Int] = Array(i, j)
        features.putScalar(indxFeatures, opT.toDouble(x))
      }
      val indxLabels:   Array[Int] = Array(i, c)
      labels.putScalar(indxLabels, 1)
    }
    new DataSet(features, labels)
  }

}
