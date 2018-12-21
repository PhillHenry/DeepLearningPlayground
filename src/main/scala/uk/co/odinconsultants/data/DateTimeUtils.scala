package uk.co.odinconsultants.data

import java.time.LocalDateTime

object DateTimeUtils {

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

}
