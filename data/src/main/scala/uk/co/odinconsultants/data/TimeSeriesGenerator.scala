package uk.co.odinconsultants.data

import java.time.LocalDateTime

import uk.co.odinconsultants.data.DateTimeUtils._

import scala.annotation.tailrec

object TimeSeriesGenerator {

  @tailrec
  def eachDay(x: LocalDateTime, end: LocalDateTime, fn: DateTimeUtils.GenerateFn, acc: Seq[Long]): Seq[Long] = {
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
