package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

import uk.co.odinconsultants.data.TimeSeriesGenerator.GenerateFn

import scala.util.Random

object TimeNoise {

  val STDDEV_MINS   = 120

  val TIMEZONE      = ZoneOffset.of("Z")

  def toLocalDateTime(l: Long): LocalDateTime = LocalDateTime.ofEpochSecond(l, 0, TIMEZONE)

  def noisyTime(offsetHour: Int): GenerateFn = { time =>
    val rndOffset = (Random.nextGaussian() * STDDEV_MINS).toLong
    val noisy     = time.plusMinutes(rndOffset)
    Seq(noisy.plusHours(offsetHour).toEpochSecond(TIMEZONE))
  }
}
