package uk.co.odinconsultants.data

import scala.util.Random

object SamplingFunctions {

  type Classified[T] = Seq[Seq[T]]

  def trainTest[T](xs: Classified[T], ratio: Double): (Seq[T], Seq[T]) = {
    val train = xs.flatMap { ys => ys.take((ratio * ys.size).toInt) }
    val test  = xs.flatMap { ys => ys.drop((ratio * ys.size).toInt) }

    (Random.shuffle(train), Random.shuffle(test))
  }

  def repeating[T](xs: Seq[T], iter: Iterator[T]): Stream[T] = Stream.cons(if (iter.hasNext) iter.next() else xs.head, repeating(xs, if (iter.hasNext) iter else xs.toIterator))

  def oversample[T](xs: Seq[(Seq[T], Double)]): Classified[T] = {
    xs.map { case (ys, w) =>
      val desired = (ys.size * w).toInt
      if (ys.size >= desired)
        ys
      else {
        val padding = repeating(ys, ys.toIterator).take(desired - ys.size)
        ys ++ padding
      }
    }
  }

}
