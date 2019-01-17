package uk.co.odinconsultants.data

import java.time.LocalDateTime

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.data.DateTimeUtils.gapsInSeconds
import uk.co.odinconsultants.maths.Stats._

@RunWith(classOf[JUnitRunner])
class ClusteredEventsDataSpec extends WordSpec with Matchers {

  import ClusteredEventsData._

  class ClusterFixture extends ClusteredEventsData{
    override def bunched2SpreadRatio: Double = 9

    override def N: Int = 100

    override def timeSeriesSize: Int = 50

    def epochSecondOf(xs: Seq[Events]): Seq[Long]
      = xs.flatMap(_._1)

    def toLocalDateTime(x: Long): LocalDateTime
      = LocalDateTime.ofEpochSecond(x, 0, DateTimeUtils.TIMEZONE)

  }

  "All data" should {
    "be red or blue" in new ClusterFixture {
      (bunched ++ spread) should have size N
      xs should have size N
    }
    "have the same series length" in new ClusterFixture {
      xs.foreach { x =>
        x._1 should have length timeSeriesSize
      }
    }
  }

  "the gaps in bunched and spread data" should {

    def meanOfStdDevOf(xs: Seq[Seq[Long]]): Double = meanOf(xs.map(stdDev(_)))
    def meanOfRanges(xs: Seq[Seq[Long]]): Double = meanOf(xs.map(x => x.max - x.min))

    "have significantly different ranges" in new ClusterFixture {
      meanOfRanges(spread.map(_._1)) should be > ( meanOfRanges(bunched.map(_._1)) * 100)
    }
    "have significantly different standard deviations" in new ClusterFixture {
      meanOfStdDevOf(spread.map(_._1)) should be > ( meanOfStdDevOf(bunched.map(_._1)) * 100)
    }
  }

}
