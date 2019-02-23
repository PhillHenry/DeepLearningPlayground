package uk.co.odinconsultants.io

import java.nio.file.Files.createTempDirectory

import org.apache.commons.io.FileUtils.forceDeleteOnExit
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import uk.co.odinconsultants.data.OfficeData
import uk.co.odinconsultants.dl4j.rnn.readers.SequenceRecordFileReader.reader

@RunWith(classOf[JUnitRunner])
class FilePersisterSpec extends WordSpec with Matchers {

  import FilePersister._

  "Sequence saved" should {
    "be readable as SequenceRecordReaderDataSetIterator" in {
      val base = createTempDirectory("FilePersisterSpec" + System.currentTimeMillis()).toFile
      forceDeleteOnExit(base)

      val data = new OfficeData(1)
      import data._
      val (featuresDir, labelsDir) = persist(base.getAbsolutePath, xs)

      val miniBatchSize = 10
      val iter          = reader(miniBatchSize, 2, xs.size - 1, featuresDir.getAbsolutePath, labelsDir.getAbsolutePath)

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
