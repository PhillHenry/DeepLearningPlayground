package uk.co.odinconsultants.data

import java.time.{LocalDateTime, ZoneOffset}

object DateTimeUtils {

  implicit val LocalDateTimeOrder = new Ordering[LocalDateTime] {
    override def compare(x: LocalDateTime, y: LocalDateTime): Int = {
      if (x.toEpochSecond(TIMEZONE)  == y.toEpochSecond(TIMEZONE)) 0 else {
        if (x.toEpochSecond(TIMEZONE) > y.toEpochSecond(TIMEZONE)) 1 else -1
      }
    }
  }

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

  type GenerateFn[T] = LocalDateTime => Seq[T]

  def gapsInSeconds(dateTimes: Seq[LocalDateTime]): Seq[Long] =
    dateTimes.sorted.sliding(2).map { xs => xs.last.toEpochSecond(TIMEZONE) - xs.head.toEpochSecond(TIMEZONE) }.toSeq

  val TIMEZONE      = ZoneOffset.of("Z")
}
