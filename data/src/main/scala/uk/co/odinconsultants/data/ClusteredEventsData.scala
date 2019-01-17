package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.DateTimeUtils._
import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.data.TimeNoise.{noisyTime, randomPoint}

import scala.util.Random
import ClusteredEventsData._

trait ClusteredEventsData extends ClassificationData[Events] {

  def bunched2SpreadRatio: Double

  def N: Int

  def timeSeriesSize: Int

  val nBlue: Int = (N / (1 + bunched2SpreadRatio)).toInt

  val nRed: Int = N - nBlue

  val BUNCHED     = 1
  val SPREAD      = 0

  val noisyFn: GenerateFn = noisyTime(0)

  val bunched: Seq[Events] = (1 to nRed).map { _ =>
    val date    = randomPoint(from, to)
    (1 to timeSeriesSize).map(_ => date).flatMap(noisyFn(_))
  }.map(_ -> BUNCHED)

  val spread: Seq[Events] = (1 to nBlue).map { _ =>
    (1 to timeSeriesSize).map(_ => randomPoint(from, to).toEpochSecond(TIMEZONE))
  }.map(_ -> SPREAD)

  val xs: Seq[Events] = Random.shuffle(bunched ++ spread)

  override val classes = Seq(bunched, spread)
}

object ClusteredEventsData {
  type Events     = (Seq[Long], Int)
}
