package uk.co.odinconsultants.dl4j.cnn

import uk.co.odinconsultants.data.MatrixData
import uk.co.odinconsultants.dl4j4s.data.DataSetShaper.OneHotMatrix2Cat

import scala.util.Random

class RawData(w: Int, h: Int, nSamples: Int = 1024, ptsPerSample: Int = 1024) {

  val randomize        = new Random()

  val noise: Seq[OneHotMatrix2Cat]      = (1 to nSamples).map { i =>
    val ranges = Seq((0, h), (0, w))
    val coords = MatrixData.randomCoords(ptsPerSample, ranges, randomize)
    (coords.map(xs => (xs.head, xs.last)), 1)
  }

  val withPattern: Seq[OneHotMatrix2Cat]   = (1 to nSamples).map { i =>
    val ranges  = Seq((0, h), (0, w))
    val s       = 4
    val pattern = (0 until w by s).zip(0 until h by s).map { case(x, y) => Seq(x, y) }
    val coords  = MatrixData.randomCoords(ptsPerSample - pattern.size, ranges, randomize) ++ pattern
    (coords.map(xs => (xs.head, xs.last)), 0)
  }

  val all           = Random.shuffle(withPattern ++ noise)
  val nTrain        = (all.length * 0.9).toInt
  val train         = all.take(nTrain)
  val test          = all.drop(nTrain)
}
