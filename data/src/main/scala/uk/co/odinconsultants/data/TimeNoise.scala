package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

import uk.co.odinconsultants.data.DateTimeUtils._

import scala.util.Random

object TimeNoise {

  val STDDEV_MINS   = 120

  def toLocalDateTime(l: Long): LocalDateTime = LocalDateTime.ofEpochSecond(l, 0, TIMEZONE)

  def noisyTime(offsetHour: Int): GenerateFn = { time =>
    val rndOffset = (Random.nextGaussian() * STDDEV_MINS).toLong
    val noisy     = time.plusMinutes(rndOffset)
    Seq(noisy.plusHours(offsetHour).toEpochSecond(TIMEZONE))
  }

  def randomPoint(from: LocalDateTime, to: LocalDateTime): LocalDateTime = {
    val max   = to.toEpochSecond(TIMEZONE)
    val min   = from.toEpochSecond(TIMEZONE)
    val range = max - min
    val rnd   = min + (Random.nextLong() % range)
    LocalDateTime.ofEpochSecond(rnd, 0, TIMEZONE)
  }
}