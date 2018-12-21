package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.data.TimeNoise.{noisyTime, randomPoint}
import uk.co.odinconsultants.data.TimeSeriesGenerator.generate

import scala.util.Random

trait ClusteredEventsData {

  def ratioRedTo1Blue: Int

  def N: Int

  def timeSeriesSize: Int

  val nBlue: Int = N / (1 + ratioRedTo1Blue)

  val nRed: Int = N - nBlue

  val RED         = 1
  val BLUE        = 0

  val red = (1 to nRed).map { _ =>
    val date = randomPoint(from, to)
    (1 to timeSeriesSize).map(_ => date).flatMap(x => noisyTime(0)(x)).map(_ -> RED)
  }

  val blue = (1 to nBlue).map { _ =>
    (1 to N).flatMap(_ => generate(start, end, noisyTime(12))).map(_ -> BLUE)
  }

  val xs = Random.shuffle(red ++ blue)
}
