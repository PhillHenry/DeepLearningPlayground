package uk.co.odinconsultants.dl4j.autoencoders

import breeze.linalg._
import breeze.linalg.svd._
import breeze.linalg._
import java.io.File
import breeze.plot._
import util.Random._

/**
  * Taken from https://gist.github.com/tyrcho/5884241
  */
object PCA extends App {

  val dimensions = 500
  val values = 2000
  val data = generateData

  val pcaRes = pca(data, 2)
  println("result pca \n" + pcaRes)

  val f1 = Figure("data")
  val f2 = Figure("pca")
  f1.subplot(0) += scatter(data(::, 0), data(::, 3), { _ => 0.1 })
  f2.subplot(0) += scatter(pcaRes(::, 0), pcaRes(::, 1), { _ => 0.1 })

  private def pca(data: DenseMatrix[Double], components: Int) = {
    val d = zeroMean(data)
    val SVD(_, _, v) = svd(d.t)
    val model = v(0 until components, ::) //top 'components' eigenvectors
    val filter = model.t * model
    filter * d
  }

  private def mean(v: Vector[Double]) = (v.valuesIterator.sum) / v.size

  private def zeroMean(m: DenseMatrix[Double]) = {
    val copy = m.copy
    for (c <- 0 until m.cols) {
      val col = copy(::, c)
      val colMean = mean(col)
      col -= colMean
    }
    //    println("data \n" + m)
    //    println("mean \n" + copy)
    copy
  }

  //val data= csvread(new File("data.csv"), skipLines = 1).t
  //  val data = DenseMatrix(
  //    (2.0, 4.0, 5.1),
  //    (1.0, 2.5, 3.5),
  //    (8.0, 3.0, 6.4),
  //    (8.0, 5.0, 6.5),
  //    (4.3, 4.5, 6.4))

  private def generateData = {
    val data = DenseMatrix.zeros[Double](values, dimensions)
    for (d <- 0 until dimensions) {
      val c1 = 2 + 2 * (nextDouble - 0.5)
      val r1 = nextDouble * 2
      val c2 = -2 + 2 * (nextDouble - 0.5)
      val r2 = nextDouble * 2
      for (v <- 0 until values / 2) {
        val vv = c1 + r1 * nextDouble
        //      println(f"a:$a b:$b d:$d v:$v vv:$vv%2f")
        data.update(v, d, vv)
      }
      for (v <- values / 2 until values) {
        val vv = c2 + r2 * nextDouble
        //      println(f"a:$a b:$b d:$d v:$v vv:$vv%2f")
        data.update(v, d, vv)
      }
    }
    data
  }
}
