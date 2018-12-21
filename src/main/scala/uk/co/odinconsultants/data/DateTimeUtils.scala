package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

object DateTimeUtils {

  val SECONDS_IN_DAY: Long = 3600 * 24

  case class DDMMYYYY(day: Int, month: Int, year: Int)

  def endOfDay(x: DDMMYYYY): LocalDateTime = {
    import x._
    LocalDateTime.of(year, month, day, 23, 59)
  }

  def startOfDay(x: DDMMYYYY): LocalDateTime = {
    import x._
    LocalDateTime.of(year, month, day, 0, 0)
  }

  type GenerateFn = LocalDateTime => Seq[Long]

  def gapsInSeconds(dateTimes: Seq[LocalDateTime]): Seq[Long] =
    dateTimes.sliding(2).map { xs => xs.last.toEpochSecond(TIMEZONE) - xs.head.toEpochSecond(TIMEZONE) }.toSeq

  val TIMEZONE      = ZoneOffset.of("Z")
}
