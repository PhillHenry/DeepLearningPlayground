package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

import uk.co.odinconsultants.data.DateTimeUtils._

import scala.util.Random

object TimeNoise {

  val STDDEV_MINS   = 120

  def toLocalDateTime(l: Long): LocalDateTime = LocalDateTime.ofEpochSecond(l, 0, TIMEZONE)

  def noisyTime(offsetHour: Int): GenerateFn[Long] = { time =>
    val rndOffset = (Random.nextGaussian() * STDDEV_MINS).toLong
    val noisy     = time.plusMinutes(rndOffset)
    Seq(noisy.plusHours(offsetHour).toEpochSecond(TIMEZONE))
  }

  def randomDateBetween(startInc: LocalDateTime, endExcl: LocalDateTime): LocalDateTime = {
    val max   = endExcl.toEpochSecond(TIMEZONE)
    val min   = startInc.toEpochSecond(TIMEZONE)
    val range = max - min
    val rnd   = min + (math.abs(Random.nextLong()) % range)
    LocalDateTime.ofEpochSecond(rnd, 0, TIMEZONE)
  }
}
