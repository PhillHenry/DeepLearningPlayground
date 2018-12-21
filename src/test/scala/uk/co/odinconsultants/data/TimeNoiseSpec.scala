package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.data.DateTimeUtils._
import uk.co.odinconsultants.data.TimeFixture._
import uk.co.odinconsultants.maths.Stats._

@RunWith(classOf[JUnitRunner])
class TimeNoiseSpec extends WordSpec with Matchers {

  import TimeNoise._

  "Random points in time" should {
    val min = startOfDay(start)
    val max = startOfDay(end)
    val n   = 1000
    "be within range and unique" in {
      val rnd = (1 to n).map { _ =>
        val time = randomPoint(min, max)
        time.isBefore(max) || time.isEqual(max)
        time.isAfter(min) || time.isEqual(min)
        time
      }.toSet
      rnd should have size n
    }

    "be distributed fairly evenly" in {
      val rnd = (1 to n).map { _ =>
        randomPoint(min, max)
      }.sorted
      val gaps                = rnd.sliding(2).map { xs => xs.last.toEpochSecond(TIMEZONE) - xs.head.toEpochSecond(TIMEZONE) }.toSeq
      val expectedAverageGap  = ((max.toEpochSecond(TIMEZONE) - min.toEpochSecond(TIMEZONE)) / n).toDouble
      meanOf(gaps) shouldBe >= (expectedAverageGap)
    }
  }

}
