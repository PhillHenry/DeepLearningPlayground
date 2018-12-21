package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

@RunWith(classOf[JUnitRunner])
class ClusteredEventsDataSpec extends WordSpec with Matchers {

  class ClusterFixture extends ClusteredEventsData{
    override def ratioRedTo1Blue: Int = 9

    override def N: Int = 100

    override def timeSeriesSize: Int = 50
  }

  "All data" should {
    "be red or blue" in new ClusterFixture {
      (red ++ blue) should have size N
      xs should have size N
    }
    "have the same series length" in new ClusterFixture {
      xs.foreach { x =>
        x._1 should have length timeSeriesSize
      }
    }
  }

}
