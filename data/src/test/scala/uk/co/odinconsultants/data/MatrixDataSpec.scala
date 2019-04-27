package uk.co.odinconsultants.data

import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class MatrixDataSpec extends WordSpec with Matchers {

  import MatrixData._

  val random = new Random()

  "Random coordinates" should {
    val n       = 100
    val range1  = (1, 100)
    val range2  = (100, 200)
    val ranges  = Seq(range1, range2)


    def checkWithinRange(x: Int, r: (Int, Int)): Unit = {
      x should be >= r._1
    }

    "be within range" in {
      val x = randomWith(range1, random)
      checkWithinRange(x, range1)
    }

    "be within ranges" in {
      val coords  = randomCoords(n, ranges, random)
      coords.foreach(xs =>
        xs should have length ranges.length
      )
    }
  }

}
