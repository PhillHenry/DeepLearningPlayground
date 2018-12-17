package uk.co.odinconsultants.data

import java.time.LocalDateTime

import org.scalatest.{Matchers, WordSpec}

import scala.collection.immutable

class OfficeDataSpec extends WordSpec with Matchers {

  import OfficeData._

  def hoursOf(xs: Seq[Long]): Seq[Int] = xs.map(toLocalDateTime).map(_.getHour)

  def meanOf(xs: Seq[Int]): Double = xs.sum.toDouble / xs.size

  def stdDev(xs: Seq[Int]): Double = {
    val mu        = meanOf(xs)
    val variance  = xs.map(x => math.pow(x - mu, 2)).sum / (xs.size - 1)
    math.pow(variance, 0.5)
  }

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

  private def checkStatsOf(stdDevMinutes: Int, xs: Seq[Int], expectedMeanHour: Double) = {
    (stdDev(xs) * 60) shouldBe stdDevMinutes.toDouble +- 10d
    meanOf(xs) shouldBe expectedMeanHour +- stdDev(xs)
  }
}
