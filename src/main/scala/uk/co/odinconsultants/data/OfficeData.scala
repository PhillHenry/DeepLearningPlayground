package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

import uk.co.odinconsultants.data.TimeSeriesGenerator.{DDMMYYYY, GenerateFn, generate}

import scala.util.Random

class OfficeData {

  import OfficeData._

  val STDDEV_MINS  = 120

  def noisyTime(offsetHour: Int): GenerateFn = { time =>
    val rndOffset = (Random.nextGaussian() * STDDEV_MINS).toLong
    val noisy     = time.plusMinutes(rndOffset)
    Seq(noisy.plusHours(offsetHour).toEpochSecond(TIMEZONE))
  }

  val start         = DDMMYYYY(1, 1, 2013)
  val end           = DDMMYYYY(1, 1, 2014)
  val NIGHT         = 1
  val DAY           = 0
  val nightTimes    = (1 to 300).flatMap(_ => generate(start, end, noisyTime(0))).map(_ -> NIGHT)
  val dayTimes      = (1 to 300).flatMap(_ => generate(start, end, noisyTime(12))).map(_ -> DAY)
  val data          = nightTimes ++ dayTimes
  val xs            = Random.shuffle(data)

}

object OfficeData {

  val TIMEZONE      = ZoneOffset.of("Z")

  def toLocalDateTime(l: Long): LocalDateTime = LocalDateTime.ofEpochSecond(l, 0, TIMEZONE)
}
