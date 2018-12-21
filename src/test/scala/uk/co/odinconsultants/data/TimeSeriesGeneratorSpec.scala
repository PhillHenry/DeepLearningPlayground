package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.data.TimeNoise.noisyTime

@RunWith(classOf[JUnitRunner])
class TimeSeriesGeneratorSpec extends WordSpec with Matchers {

  import TimeSeriesGenerator._

  "one point per day" should {
    val points = generate(DDMMYYYY(1, 1, 2019), DDMMYYYY(1, 1, 2020), noisyTime(0))
    "generate data" in {
      points should not be None
    }
    "generate 365 points" in {
      points.get should have size 365
    }
    "span nearly a year" in {
      val start = points.get.min
      val end   = points.get.max
      withClue(s"start = ${new java.util.Date(start * 1000)}, end = ${new java.util.Date(end * 1000)}\n") {
        (end - start) shouldBe > (363 * 24 * 3600L)
      }
    }
  }

}
