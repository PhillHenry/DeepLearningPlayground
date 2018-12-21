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
import uk.co.odinconsultants.io.FilePersister.persist
import uk.co.odinconsultants.data.{ClusteredEventsData, OfficeData}
import uk.co.odinconsultants.dl4j.rnn.readers.SequenceRecordFileReader.reader

import scala.collection.JavaConverters._


object TimeSeries {

  def process(): MultiLayerNetwork = {
    val data          = new ClusteredEventsData {
      override def ratioRedTo1Blue: Int = 1

      override def N: Int = 600

      override def timeSeriesSize: Int = 50
    }
    val trainSize     = (data.xs.size * 0.9).toInt
    val train         = data.xs.take(trainSize)
    val test          = data.xs.drop(trainSize)
    val nClasses      = 2
    val nIn           = 1
    val m             = model(nIn, nClasses)
    val nEpochs       = 10

    val dir           = TimeSeries.getClass.getSimpleName
    val base          = createTempDirectory(dir).toFile
    forceDeleteOnExit(base)
    val (trainFeaturesDir, trainLabelsDir)  = persist(dir + File.separator + "train", train)
    val (testFeaturesDir, testLabelsDir)    = persist(dir + File.separator + "test",  test)
    val miniBatchSize = 10
    val trainIter     = reader(miniBatchSize, nClasses, train.size - 1, trainFeaturesDir.getAbsolutePath, trainLabelsDir.getAbsolutePath)
    val testIter      = reader(miniBatchSize, nClasses, test.size - 1,  testFeaturesDir.getAbsolutePath,  testLabelsDir.getAbsolutePath)

    val normalizer = new NormalizerStandardize
    normalizer.fit(trainIter) //Collect training data statistics

    trainIter.reset()

    //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    val str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f"
    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(trainIter)
      val evaluation: Evaluation = m.evaluate(testIter)
      val f1: Double = evaluation.f1
      val accuracy: Double = evaluation.accuracy
      println(str.format(i, accuracy, f1))

      testIter.reset()
      trainIter.reset()
    }

    val evaluation: Evaluation = m.evaluate(testIter)
    println("Accuracy: "+evaluation.accuracy)
    println("Accuracy: "+evaluation.stats())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())

    m
  }

  /**
    * Stolen from https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
    *
    * hidden layer of 100 => "Warning: 1 class was never predicted by the model and was excluded from average precision"
    */
  def model( inN: Int, nClasses: Int): MultiLayerNetwork = {
    val tbpttLength = 100
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123) //Random number generator seed for improved repeatability. Optional.
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.005))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(10).nOut(nClasses).build())
      .build();

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1000 / tbpttLength))
    model
  }

  def main(xs: Array[String]): Unit = {
    println(process())
  }

}
