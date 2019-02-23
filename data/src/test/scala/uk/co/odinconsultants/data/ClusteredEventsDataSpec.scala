package uk.co.odinconsultants.data

import java.time.LocalDateTime

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.maths.Stats._

@RunWith(classOf[JUnitRunner])
class ClusteredEventsDataSpec extends WordSpec with Matchers {

  class ClusterFixture {

    val bunched2SpreadRatio:  Double              = 9
    val N:                    Int                 = 100
    val timeSeriesSize:       Int                 = 50
    val data:                 ClusteredEventsData = new ClusteredEventsData(bunched2SpreadRatio, N, timeSeriesSize, 1)

  }

  "Data generated with the same seed" should {
    "be the same" in {
      val data1 = new ClusterFixture
      val data2 = new ClusterFixture

      data1.data.classes shouldBe data2.data.classes
    }
  }

  "All data" should {
    "be red or blue" in new ClusterFixture {
      import data._
      (bunched ++ spread) should have size N
      xs should have size N
    }
    "have the same series length" in new ClusterFixture {
      import data._
      xs.foreach { x =>
        x._1 should have length timeSeriesSize
      }
    }
  }

  "the gaps in bunched and spread data" should {

    def meanOfStdDevOf(xs: Seq[Seq[Long]]): Double = meanOf(xs.map(stdDev(_)))
    def meanOfRanges(xs: Seq[Seq[Long]]): Double = meanOf(xs.map(x => x.max - x.min))

    "have significantly different ranges" in new ClusterFixture {
      import data._
      meanOfRanges(spread.map(_._1)) should be > ( meanOfRanges(bunched.map(_._1)) * 100)
    }
    "have significantly different standard deviations" in new ClusterFixture {
      import data._
      meanOfStdDevOf(spread.map(_._1)) should be > ( meanOfStdDevOf(bunched.map(_._1)) * 100)
    }
  }

}
