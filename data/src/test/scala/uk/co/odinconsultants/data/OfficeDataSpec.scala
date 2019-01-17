package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.data.TimeNoise._
import uk.co.odinconsultants.maths.Stats._

@RunWith(classOf[JUnitRunner])
class OfficeDataSpec extends WordSpec with Matchers {

  def hoursOf(xs: Seq[Long]): Seq[Int] = xs.map(toLocalDateTime).map(_.getHour)

  "Night and day" should {
    "be significantly different" in {
      val data = new OfficeData
      import data._
      val ds = xs.filter(_._2 == DAY).map(_._1).flatMap(hoursOf)
      val ns = xs.filter(_._2 == NIGHT).map(_._1).flatMap(hoursOf).map(x => if (x > 12) x - 24 else x)

      checkStatsOf(STDDEV_MINS, ds, 12d)
      checkStatsOf(STDDEV_MINS, ns, 0d)
    }
  }

  private def checkStatsOf(stdDevMinutes: Double, xs: Seq[Int], expectedMeanHour: Double) = {
    val sd = stdDev(xs)
    val mu = meanOf(xs)
    println(s"mean = $mu, std dev = $sd")
    (sd * 60) shouldBe stdDevMinutes +- 10d
    mu shouldBe expectedMeanHour +- sd
  }
}
