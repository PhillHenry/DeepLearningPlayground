package uk.co.odinconsultants.data

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

@RunWith(classOf[JUnitRunner])
class SamplingFunctionsSpec extends WordSpec with Matchers {

  import SamplingFunctions._

  trait ClassifiedData {
    val xs: Seq[String]   = Array.fill(10)("x")
    val ys: Seq[String]   = Array.fill(100)("y")
    val zs: Seq[String]   = Array.fill(1000)("z")

    val classified: Seq[Seq[String]]        = Seq(xs, ys, zs)
  }

  def numberOf(xs: Seq[String], x: String): Int = xs.filter(_ == x).size

  "splitting" should {
    "give equal proportions of classes" in new ClassifiedData {
      val ratio             = 0.9
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

  "infinite stream" should {
    "repeat sequences" in {
      val xs      = (1 to 10)
      val stream  = repeating(xs, xs.toIterator)
      val n       = 3
      val result  = stream.take(xs.size * n)
      result should have size (n * xs.size)
      xs.foreach { x =>
        result.filter(_ == x) should have size n
      }
    }
  }

  "oversampling" should {
    "increase size of minor class" in new ClassifiedData {
      val weights:        Seq[Double]                 = Seq(1d, 1d, 10d)
      val class2Weights:  Seq[(Seq[String], Double)]  = classified.zip(weights)
      val oversampled                                 = oversample(class2Weights)
      oversampled.zip(class2Weights).foreach { case (actual, (expected, w)) =>
        val sample = actual.head
        withClue(s"sequence of '$sample'") {
          actual should have size (expected.size * w).toInt
        }
      }
    }
  }

}
