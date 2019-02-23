package uk.co.odinconsultants.data

import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.data.TimeNoise.noisyTime
import uk.co.odinconsultants.data.TimeSeriesGenerator.generate

import scala.util.Random

class OfficeData(seed: Int) {

  val random:     Random                = new Random(seed)
  val NIGHT                             = 1
  val DAY                               = 0
  val N                                 = 600
  val nightTimes: Seq[(Seq[Long], Int)] = (1 to N).flatMap(_ => generate(start, end, noisyTime(0, random))).map(_ -> NIGHT)
  val dayTimes:   Seq[(Seq[Long], Int)] = (1 to N).flatMap(_ => generate(start, end, noisyTime(12, random))).map(_ -> DAY)
  val data:       Seq[(Seq[Long], Int)] = nightTimes ++ dayTimes
  val xs:         Seq[(Seq[Long], Int)] = random.shuffle(data) //data.map { case(ys, i) => random.shuffle(ys) -> i }

}
