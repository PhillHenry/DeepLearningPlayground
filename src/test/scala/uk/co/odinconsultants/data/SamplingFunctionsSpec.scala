package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

@RunWith(classOf[JUnitRunner])
class SamplingFunctionsSpec extends WordSpec with Matchers {

  import SamplingFunctions._

  def numberOf(xs: Seq[String], x: String): Int = xs.filter(_ == x).size

  "splitting" should {
    "give equal proportions of classes" in {
      val xs: Seq[String]   = Array.fill(10)("x")
      val ys: Seq[String]   = Array.fill(100)("y")
      val zs: Seq[String]   = Array.fill(1000)("z")
      val ratio             = 0.9

      val classified        = Seq(xs, ys, zs)
      val (train, test)     = trainTest(classified, 0.9)

      numberOf(test, "x") shouldBe 1
      numberOf(test, "y") shouldBe 10
      numberOf(test, "z") shouldBe 100
      test should have size 111

      numberOf(train, "x") shouldBe 9
      numberOf(train, "y") shouldBe 90
      numberOf(train, "z") shouldBe 900
      train should have size 999
    }
  }

}
