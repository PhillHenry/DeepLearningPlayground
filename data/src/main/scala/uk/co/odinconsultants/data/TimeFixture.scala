package uk.co.odinconsultants.data

import java.time.LocalDateTime

import uk.co.odinconsultants.data.DateTimeUtils._

import scala.util.Random

object TimeFixture {

  val start         = DDMMYYYY(1, 1, 2013)
  val end           = DDMMYYYY(1, 1, 2014)

  val from:   LocalDateTime = startOfDay(start)
  val to:     LocalDateTime = startOfDay(end)
  val random: Random        = new Random(1)

}
