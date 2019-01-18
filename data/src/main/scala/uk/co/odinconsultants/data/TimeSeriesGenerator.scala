package uk.co.odinconsultants.data

import java.time.LocalDateTime

import uk.co.odinconsultants.data.DateTimeUtils._

import scala.annotation.tailrec

object TimeSeriesGenerator {

  @tailrec
  def eachDay[T](x: LocalDateTime, end: LocalDateTime, fn: GenerateFn[T], acc: Seq[T]): Seq[T] = {
    if (!x.isBefore(end))
      acc
    else
      eachDay(x.plusDays(1), end, fn, acc ++ fn(x))
  }

  def generate[T](fromInclusive: DDMMYYYY, toExclusive: DDMMYYYY, fn: GenerateFn[T]): Option[Seq[T]] = {
    val from  = startOfDay(fromInclusive)
    val to    = startOfDay(toExclusive)
    if (!to.isAfter(from))
      None
    else
      Some(eachDay(from, to, fn, Seq()))
  }

}
