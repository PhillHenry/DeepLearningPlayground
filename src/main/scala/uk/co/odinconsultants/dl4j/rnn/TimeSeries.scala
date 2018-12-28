package uk.co.odinconsultants.dl4j.rnn

import java.io.File
import java.nio.file.Files.createTempDirectory

import org.apache.commons.io.FileUtils.forceDeleteOnExit
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
import uk.co.odinconsultants.io.FilePersister.persist
import uk.co.odinconsultants.data.{ClusteredEventsData, OfficeData}
import uk.co.odinconsultants.dl4j.rnn.readers.SequenceRecordFileReader.reader

import scala.collection.JavaConverters._


object TimeSeries {

  def process(): MultiLayerNetwork = {
    val data          = new ClusteredEventsData {
      override def bunched2SpreadRatio: Double = 0.01

      override def N: Int = 6000

      override def timeSeriesSize: Int = 50
    }
    val trainSize     = (data.xs.size * 0.9).toInt
    val train         = data.xs.take(trainSize)
    val test          = data.xs.drop(trainSize)
    val nClasses      = 2
    val nIn           = 1
    val m             = model(nIn, nClasses)
    val nEpochs       = 5

    val jTrain        = to3DDataset(train, nClasses, data.timeSeriesSize, nIn)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)
    val testDataSets  = test.map(x => to3DDataset(Seq(x), nClasses, data.timeSeriesSize, nIn)).toList.asJava
    val testIter      = new ListDataSetIterator(testDataSets)

    val normalizer = new NormalizerStandardize
    normalizer.fit(trainIter) //Collect training data statistics

    trainIter.reset()

    //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(trainIter)
      val evaluation: Evaluation = m.evaluate(testIter)
      val f1: Double = evaluation.f1
      println("Test set evaluation at epoch %d: Accuracy = %.2f, Precision = %.2f, F1 = %.2f".format(i, evaluation.accuracy, evaluation.precision(), f1))

      testIter.reset()
      trainIter.reset()
    }

    val evaluation: Evaluation = m.evaluate(testIter)
    println("Accuracy: "+evaluation.accuracy)
    println("Stats: "+evaluation.stats())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())

    m
  }

  /*
  Epochs = 5; N = 6000, time series size = 50; spread:clustered = 100:1
Accuracy: 0.9929666666666667
Stats:

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9930
 Precision:       0.9586
 Recall:          0.7572
 F1 Score:        0.6613
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
     0     1
-------------
 29583    17 | 0 = 0
   194   206 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.9237668161434978
Recall: 0.515


new LossNegativeLogLikelihood(Nd4j.create(Array(0.01f, 1f)))
Results:
Accuracy: 0.9850333333333333
Stats:

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9850
 Precision:       0.6290
 Recall:          0.8311
 F1 Score:        0.3755
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
     0     1
-------------
 29416   384 | 0 = 0
    65   135 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.26011560693641617
Recall: 0.675


Now with new LossNegativeLogLikelihood(Nd4j.create(Array(0.1f, 1f)))
Results:
Accuracy: 0.996
Stats:

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9960
 Precision:       0.8655
 Recall:          0.6564
 F1 Score:        0.4393
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
     0     1
-------------
 29833    17 | 0 = 0
   103    47 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.734375
Recall: 0.31333333333333335


new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))
Results:
Accuracy: 0.9760666666666666
Stats:

========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0.9761
 Precision:       0.6592
 Recall:          0.9526
 F1 Score:        0.4751
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
     0     1
-------------
 28957   693 | 0 = 0
    25   325 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.3192534381139489
Recall: 0.9285714285714286

   */

  type Series2Cat = (Seq[Long], Int)

  /**
    * Aha! Was the victim of this bug: https://github.com/deeplearning4j/dl4j-examples/issues/779
    */
  def to3DDataset(s2cs: Seq[Series2Cat], nClasses: Int, seriesLength: Int, nIn: Int): DataSet = {
    val n         = s2cs.size
    val features  = Nd4j.zeros(n, nIn, seriesLength)
    val labels    = Nd4j.zeros(n, nClasses, seriesLength)

    s2cs.zipWithIndex.foreach { case ((xs, c), i) =>
      xs.zipWithIndex.foreach { case (x, j) =>
        val indxFeatures: Array[Int] = Array(i, 0, j)
        features.putScalar(indxFeatures, x)
        val indxLabels:   Array[Int] = Array(i, c, j)
        labels.putScalar(indxLabels, 1)
      }
    }
    new DataSet(features, labels)
  }


  /**
    * Stolen from https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
    *
    */
  def model( inN: Int, nClasses: Int): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123) //Random number generator seed for improved repeatability. Optional.
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.005))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(10).nOut(nClasses).lossFunction(new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))).build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))
    model
  }

  def main(xs: Array[String]): Unit = {
    println(process())
  }

}
