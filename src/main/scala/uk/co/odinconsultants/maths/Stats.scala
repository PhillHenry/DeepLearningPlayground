package uk.co.odinconsultants.maths

object Stats {

  def meanOf[T : Numeric](xs: Seq[T]): Double =
    implicitly[Numeric[T]].toDouble(xs.sum) / xs.size

  def stdDev[T : Numeric](xs: Seq[T]): Double = {
    val mu        = meanOf(xs)
    val variance  = xs.map { x =>
      math.pow(implicitly[Numeric[T]].toDouble(x) - mu, 2)
    }.sum / (xs.size - 1)
    math.pow(variance, 0.5)
  }


}
