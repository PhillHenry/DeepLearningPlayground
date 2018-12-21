package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.TimeNoise.noisyTime
import uk.co.odinconsultants.data.TimeSeriesGenerator.{DDMMYYYY, generate}

import scala.util.Random

class OfficeData {

  val start         = DDMMYYYY(1, 1, 2013)
  val end           = DDMMYYYY(1, 1, 2014)
  val NIGHT         = 1
  val DAY           = 0
  val N             = 600
  val nightTimes    = (1 to N).flatMap(_ => generate(start, end, noisyTime(0))).map(_ -> NIGHT)
  val dayTimes      = (1 to N).flatMap(_ => generate(start, end, noisyTime(12))).map(_ -> DAY)
  val data          = nightTimes ++ dayTimes
  val xs            = Random.shuffle(data)

}
