package uk.co.odinconsultants.data

import java.nio.file.Files.createTempDirectory

import org.apache.commons.io.FileUtils.forceDeleteOnExit
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.scalatest.{Matchers, WordSpec}

class FilePersisterSpec extends WordSpec with Matchers {

  import FilePersister._

  "Sequence saved" should {
    "be readable as SequenceRecordReaderDataSetIterator" in {
      val base = createTempDirectory("FilePersisterSpec" + System.currentTimeMillis()).toFile
      forceDeleteOnExit(base)

      val data = new OfficeData
      import data._
      val (featuresDirTrain, labelsDirTrain) = persist(base.getAbsolutePath, xs)

      val features        = new CSVSequenceRecordReader
      features.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath + "/%d.csv", 0, xs.size - 1))
      val labels          = new CSVSequenceRecordReader
      labels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath + "/%d.csv", 0, xs.size - 1))
      val miniBatchSize   = 10
      val numLabelClasses = 2
      val iter            = new SequenceRecordReaderDataSetIterator(features, labels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

      iter.hasNext shouldBe true
      var count = 0
      while (iter.hasNext) {
        iter.next() should not be null
        count = count + 1
      }

      count shouldBe (xs.size / miniBatchSize)
    }
  }

}
