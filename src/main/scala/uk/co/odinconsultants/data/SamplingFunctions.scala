package uk.co.odinconsultants.data

object SamplingFunctions {

  def trainTest[T](xs: Seq[Seq[T]], ratio: Double): (Seq[T], Seq[T]) = {
    val train = xs.flatMap { ys => ys.take((ratio * ys.size).toInt) }
    val test  = xs.flatMap { ys => ys.drop((ratio * ys.size).toInt) }

    (train, test)
  }

}
