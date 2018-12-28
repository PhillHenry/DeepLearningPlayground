package uk.co.odinconsultants.data

trait ClassificationData[T] {

  val classes: Seq[Seq[T]]

}
