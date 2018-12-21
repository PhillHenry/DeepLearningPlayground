package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.DateTimeUtils._
import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.data.TimeNoise.{noisyTime, randomPoint}

import scala.util.Random

trait ClusteredEventsData {

  def ratioRedTo1Blue: Int

  def N: Int

  def timeSeriesSize: Int

  val nBlue: Int = N / (1 + ratioRedTo1Blue)

  val nRed: Int = N - nBlue

  val RED         = 1
  val BLUE        = 0

  type Events     = (Seq[Long], Int)

  val noisyFn: GenerateFn = noisyTime(0)

  val bunched: Seq[Events] = (1 to nRed).map { _ =>
    val date    = randomPoint(from, to)
    (1 to timeSeriesSize).map(_ => date).flatMap(noisyFn(_))
  }.map(_ -> RED)

  val spread: Seq[Events] = (1 to nBlue).map { _ =>
    (1 to timeSeriesSize).map(_ => randomPoint(from, to).toEpochSecond(TIMEZONE))
  }.map(_ -> BLUE)

  val xs: Seq[Events] = Random.shuffle(bunched ++ spread)
}
