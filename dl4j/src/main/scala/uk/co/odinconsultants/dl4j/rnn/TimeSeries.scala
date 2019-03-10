package uk.co.odinconsultants.dl4j.rnn

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
import uk.co.odinconsultants.data.ClusteredEventsData
import uk.co.odinconsultants.data.ClusteredEventsData.ClassifiedSample
import uk.co.odinconsultants.data.SamplingFunctions.{oversample, trainTest}
import uk.co.odinconsultants.dl4j.MultiDimension._

import scala.collection.JavaConverters._


object TimeSeries {

  def process(): MultiLayerNetwork = {
    val bunched2SpreadRatio = 0.01
    val N                   = 6000
    val timeSeriesSize      = 50
    val data                = new ClusteredEventsData(bunched2SpreadRatio, N, timeSeriesSize, 1)
    import data._
    /*
No oversampling:
     0     1
-------------
 28868   832 | 0 = 0
     3   297 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.2630646589902569
Recall: 0.99


With (bunched, 10d), (spread, 1)
     0     1
-------------
 28283  1417 | 0 = 0
     0  3000 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.6791940230925968
Recall: 1.0

With (bunched, 100d), (spread, 1)
     0     1
-------------
 28037  1663 | 0 = 0
     0 30000 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Precision: 0.9474781290465212
Recall: 1.0
     */

    val toOver: Seq[(Seq[ClassifiedSample], Double)] = Seq((bunched, 10d), (spread, 1))
    val oversampled   = oversample(toOver)
    val (train, test) = trainTest(oversampled, 0.9)
    val nClasses      = 2
    val nIn           = 1
    val m             = model(nIn, nClasses)
    val nEpochs       = 5

    val jTrain        = to3DDataset(train, nClasses, timeSeriesSize, nIn)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)
    val testDataSets  = test.map(x => to3DDataset(Seq(x), nClasses, timeSeriesSize, nIn)).toList.asJava
    val testIter      = new ListDataSetIterator(testDataSets)

    val normalizer = new NormalizerStandardize
    normalizer.fit(trainIter)

    trainIter.reset()

    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(trainIter)
      val evaluation: Evaluation = m.evaluate(testIter)
      val f1: Double = evaluation.f1
      println("Test set evaluation at epoch %d: Accuracy = %.2f, Precision = %.2f, Recall = %.2f, F1 = %.2f".format(i, evaluation.accuracy, evaluation.precision(), evaluation.recall(), f1))

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

  /**
    * Stolen from https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
    *
    */
  def model( inN: Int, nClasses: Int): MultiLayerNetwork = {
    val nHidden = 20
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123) //Random number generator seed for improved repeatability. Optional.
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.005))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(nHidden).build())
      .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).nIn(nHidden).nOut(nClasses).lossFunction(new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))).build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))


    /* see https://deeplearning4j.org/docs/latest/deeplearning4j-nn-visualization */
    import org.deeplearning4j.ui.api.UIServer
    import org.deeplearning4j.ui.stats.StatsListener
    import org.deeplearning4j.ui.storage.InMemoryStatsStorage
    //Initialize the user interface backend
    val uiServer = UIServer.getInstance

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    model.setListeners(new StatsListener(statsStorage))

    model
  }

  def main(xs: Array[String]): Unit = {
    println(process())
  }

}
