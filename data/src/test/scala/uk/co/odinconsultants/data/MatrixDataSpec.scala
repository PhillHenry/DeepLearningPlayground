package uk.co.odinconsultants.data

import org.scalatest.{Matchers, WordSpec}

class MatrixDataSpec extends WordSpec with Matchers {

  import MatrixData._

  "Random coordinates" should {
    val n       = 100
    val range1  = (1, 100)
    val range2  = (100, 200)
    val ranges  = Seq(range1, range2)
    val coords  = randomCoords(n, ranges)

    "be within range" in {
      coords.foreach(xs =>
        xs should have length ranges.length
      )
    }
  }

}
