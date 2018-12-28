package uk.co.odinconsultants.dl4j.rnn

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.nd4j.evaluation.classification.Evaluation
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
import uk.co.odinconsultants.data.ClusteredEventsData

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
