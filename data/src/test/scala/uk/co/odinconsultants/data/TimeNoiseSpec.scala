package uk.co.odinconsultants.data

import java.time.LocalDateTime

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
    val n   = 100
    "be within range and unique" in {
      val rnd = (1 to n).map { _ =>
        val time = randomDateBetween(from, to)
        time.isBefore(to) || time.isEqual(to)
        time.isAfter(from) || time.isEqual(from)
        time
      }.toSet
      rnd should have size n
    }

    "be distributed fairly evenly" in {
      val rnd = (1 to n).map { _ =>
        randomDateBetween(from, to)
      }.sorted
      val expectedNumGaps     = n - 1
      val gaps                = gapsInSeconds(rnd)
      gaps should have length expectedNumGaps
      val duration            = to.toEpochSecond(TIMEZONE) - from.toEpochSecond(TIMEZONE)
      val expectedAverageGap  = duration.toDouble / expectedNumGaps
      meanOf(gaps) <= (expectedAverageGap) // unless first = from and last = to
    }
  }

}
