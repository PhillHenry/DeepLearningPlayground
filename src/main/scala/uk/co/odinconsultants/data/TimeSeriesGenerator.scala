package uk.co.odinconsultants.data

import java.time.LocalDateTime

import scala.annotation.tailrec

object TimeSeriesGenerator {

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

  @tailrec
  def eachDay(x: LocalDateTime, end: LocalDateTime, fn: GenerateFn, acc: Seq[Long]): Seq[Long] = {
    if (!x.isBefore(end))
      acc
    else
      eachDay(x.plusDays(1), end, fn, acc ++ fn(x))
  }

  def generate(fromInclusive: DDMMYYYY, toExclusive: DDMMYYYY, fn: GenerateFn): Option[Seq[Long]] = {
    val from  = startOfDay(fromInclusive)
    val to    = startOfDay(toExclusive)
    if (!to.isAfter(from))
      None
    else
      Some(eachDay(from, to, fn, Seq()))
  }

}
