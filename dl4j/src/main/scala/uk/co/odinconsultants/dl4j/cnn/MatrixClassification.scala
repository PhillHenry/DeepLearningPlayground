package uk.co.odinconsultants.dl4j.cnn

import java.util
import java.util.{HashMap, Map}

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{MapSchedule, ScheduleType}
import uk.co.odinconsultants.data.MatrixData
import uk.co.odinconsultants.dl4j.autoencoders.AnomalyDetection
import uk.co.odinconsultants.dl4j4s.data.DataSetShaper
import uk.co.odinconsultants.dl4j4s.data.DataSetShaper.OneHotMatrix2Cat

import scala.util.Random

object MatrixClassification {
  def main(args: Array[String]): Unit = {
    val h                     = 100
    val w                     = 100
    val (testIter, trainIter) = testTrain(w, h)
    val m                     = model(h, w)
    val nEpochs               = 5
    (1 to nEpochs).foreach { e =>
      println(s"Epoch # $e")
      m.fit(trainIter)
      trainIter.reset()
    }

    val evaluation: Evaluation = m.evaluate(testIter)
    println("Accuracy: "+evaluation.accuracy)
    println("Stats: "+evaluation.stats())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())
  }

  def rawTestTrain(w: Int, h: Int): (Seq[OneHotMatrix2Cat], Seq[OneHotMatrix2Cat]) = {
    val all           = randPattern(h, w)

    val nTrain        = (all.length * 0.9).toInt
    val train         = all.take(nTrain)
    val test          = all.drop(nTrain)
    (test, train)
  }

  type Data = ListDataSetIterator[DataSet]

  def testTrain(w: Int, h: Int): (Data, Data) = {
    val (test, train) = rawTestTrain(w, h)
    val shaper        = new DataSetShaper[Double]
    val nClasses      = 2
    val dsTrain       = shaper.to4DDataset(train, nClasses, w, h)
    val dsTest        = shaper.to4DDataset(test, nClasses, w, h)
    val batchSize     = 32
    val trainIter     = new ListDataSetIterator(dsTrain.batchBy(1), batchSize)
    val testIter      = new ListDataSetIterator(dsTest.batchBy(1), batchSize)
    (testIter, trainIter)
  }

  def randPattern(h: Int, w: Int): Seq[OneHotMatrix2Cat] = {
    val nSamples      = 1024
    val ptsPerSample  = 1024
    val random        = new Random()
    val m2csRand: Seq[OneHotMatrix2Cat]      = (1 to nSamples).map { i =>
      val ranges = Seq((0, h), (0, w))
      val coords = MatrixData.randomCoords(ptsPerSample, ranges, random)
      (coords.map(xs => (xs.head, xs.last)), 1)
    }
    val m2csPattern: Seq[OneHotMatrix2Cat]   = (1 to nSamples).map { i =>
      val ranges  = Seq((0, h), (0, w))
      val s       = 4
      val pattern = (0 until w by s).zip(0 until h by s).map { case(x, y) => Seq(x, y) }
      val coords  = MatrixData.randomCoords(ptsPerSample - pattern.size, ranges, random) ++ pattern
      (coords.map(xs => (xs.head, xs.last)), 0)
    }
    Random.shuffle(m2csPattern ++ m2csRand)
  }

  /**
    * Stolen from MnistClassifier in dl4j-examples
    */
  def model(height: Int, width: Int): MultiLayerNetwork = {
    val seed = 1L
    val learningRateSchedule: util.Map[java.lang.Integer, java.lang.Double] = new util.HashMap[java.lang.Integer, java.lang.Double]
    val x = 100
    learningRateSchedule.put(0, 0.06 / x)
    learningRateSchedule.put(200, 0.05 / x)
    learningRateSchedule.put(600, 0.028 / x)
    learningRateSchedule.put(800, 0.0060 / x)
    learningRateSchedule.put(1000, 0.001 / x)
    val updater = new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule))
    val channels = 1
    val outputNum = 2

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005).updater(updater)
      .weightInit(WeightInit.XAVIER).list
      .layer(new ConvolutionLayer.Builder(5, 5).nIn(channels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build)
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut // nIn need not specified in later layers
        (50).activation(Activation.IDENTITY).build)
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build)
      .layer(
        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nOut(outputNum).activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(height, width, channels))
      .build // InputType.convolutional for normal image


    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()

    AnomalyDetection.uiServerListensTo(net)
    net.setListeners(new ScoreIterationListener(10))
    net
  }

}
