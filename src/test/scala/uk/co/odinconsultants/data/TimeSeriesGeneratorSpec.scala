package uk.co.odinconsultants.data

import org.scalatest.{Matchers, WordSpec}

class TimeSeriesGeneratorSpec extends WordSpec with Matchers {

  import TimeSeriesGenerator._

  "one point per day" should {
    "generate 365 points in a year" in {
      val points = generate(DDMMYYYY(1, 1, 2019), DDMMYYYY(1, 1, 2020), _ => Seq(1L))
      points should not be None
      points.get should have size 365
    }
  }

}
