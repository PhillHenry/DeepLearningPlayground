package uk.co.odinconsultants.dl4j.rnn.readers

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END

object SequenceRecordFileReader {

  def reader(miniBatchSize:   Int,
             numLabelClasses: Int,
             maxIdxInclusive: Int,
             featuresDir:     String,
             labelsDir:       String): SequenceRecordReaderDataSetIterator = {
    val features        = new CSVSequenceRecordReader
    features.initialize(new NumberedFileInputSplit(featuresDir + "/%d.csv", 0, maxIdxInclusive))
    val labels          = new CSVSequenceRecordReader
    labels.initialize(new NumberedFileInputSplit(labelsDir + "/%d.csv", 0, maxIdxInclusive))
    new SequenceRecordReaderDataSetIterator(features, labels, miniBatchSize, numLabelClasses, false, ALIGN_END)
  }

}
