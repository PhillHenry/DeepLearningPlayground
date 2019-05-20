package uk.co.odinconsultants.dl4j.cnn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot.{Figure, scatter}
import breeze.plot._

object RawDataPlot {

  def main(args: Array[String]): Unit = {
    val nCols = 100
    val nRows = 100
    val raw = new RawData(nCols, nRows)
    import raw._
    val f1 = Figure("Random")
    val f2 = Figure("Pattern")
    val rand = noise.head._1
    val pattern = withPattern.head._1

    def toDenseMatrix(xs: Seq[(Int, Int)], fn: ((Int, Int)) => Int): DenseVector[Int] =
      new DenseVector(xs.map(fn).toArray)

    f1.subplot(0) += scatter(toDenseMatrix(rand, _._1),     toDenseMatrix(rand, _._2),    { _ => 0.1 })
    f2.subplot(0) += scatter(toDenseMatrix(pattern, _._1),  toDenseMatrix(pattern, _._2), { _ => 0.1 })
  }

}
