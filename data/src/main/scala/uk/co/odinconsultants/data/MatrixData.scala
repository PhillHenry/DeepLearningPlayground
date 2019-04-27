package uk.co.odinconsultants.data

import scala.util.Random

object MatrixData {

  type Range = (Int, Int)

//  case class RandFn[T](seed: RandFn[T]) {
//    def next: (T, RandFn[T])
//  }

  def randomWith(r: Range, random: Random): Int = {
    val diff = r._2 - r._1
    (random.nextDouble() * diff).toInt
  }

  def randomCoords(n: Int, ranges: Seq[Range], random: Random): Seq[Seq[Int]] = {
    (1 to n).map { _ =>
      ranges.map(r => randomWith(r, random))
    }

  }

}
