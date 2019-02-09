package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.DateTimeUtils._
import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.data.TimeNoise.{noisyTime, randomDateBetween}

import scala.util.Random
import ClusteredEventsData._

trait ClusteredEventsData extends ClassificationData[ClassifiedSample] {

  def bunched2SpreadRatio: Double

  def N: Int

  def timeSeriesSize: Int

  val nBlue: Int = (N / (1 + bunched2SpreadRatio)).toInt

  val nRed: Int = N - nBlue

  val BUNCHED     = 1
  val SPREAD      = 0

  val noisyFn: GenerateFn[Long] = noisyTime(0)

  val bunched: Seq[ClassifiedSample] = (1 to nRed).map { _ =>
    val date    = randomDateBetween(from, to)
    (1 to timeSeriesSize).flatMap(_ => noisyFn(date))
  }.map(_ -> BUNCHED)

  val spread: Seq[ClassifiedSample] = (1 to nBlue).map { _ =>
    (1 to timeSeriesSize).map(_ => randomDateBetween(from, to).toEpochSecond(TIMEZONE))
  }.map(_ -> SPREAD)

  val xs: Seq[ClassifiedSample] = Random.shuffle(bunched ++ spread)

  override val classes = Seq(bunched, spread)
}

object ClusteredEventsData {
  type ClassifiedSample     = (Seq[Long], Int)
}
