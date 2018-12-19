package uk.co.odinconsultants.dl4j.rnn

import java.time.ZoneOffset
import java.util.{List => JList}

import org.datavec.api.records.reader.impl.inmemory.InMemorySequenceRecordReader
import org.datavec.api.writable.{IntWritable, LongWritable, Writable}
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import uk.co.odinconsultants.data.OfficeData
import uk.co.odinconsultants.data.TimeSeriesGenerator._

import scala.collection.JavaConverters._
import scala.util.Random


object TimeSeries {

  def process(): MultiLayerNetwork = {
    val data          = new OfficeData
    val trainSize     = (data.xs.size * 0.9).toInt
    val train         = data.xs.take(trainSize)
    val test          = data.xs.drop(trainSize)
    val nClasses      = 2
    val seriesLength  = train.head._1.length
    val nIn           = 1
    val jTrain        = to3DDataset(train, nClasses, seriesLength, nIn: Int)
    val m             = model(nIn, nClasses)
    val nEpochs = 20
    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(jTrain)
    }

    val testDataSets = test.map(x => to3DDataset(Seq(x), nClasses, seriesLength, nIn)).toList.asJava
    val iter = new ListDataSetIterator(testDataSets)

    val evaluation: Evaluation = m.evaluate(iter)
    println("Accuracy: "+evaluation.accuracy)
    println("Accuracy: "+evaluation.stats())
    println("Precision: "+evaluation.precision)
    println("Recall: "+evaluation.recall)

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
//    features.setOrder('f')
//    labels.setOrder('f')
    new DataSet(features, labels)
  }

  /**
    * Stolen from https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
    */
  def model( inN: Int, nClasses: Int): MultiLayerNetwork = {
    val tbpttLength = 50
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.05, 1))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(inN).nOut(100).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(100).nOut(nClasses).build())
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
//      .pretrain(false).backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(20))
    model
  }

  def main(xs: Array[String]): Unit = {
    println(process())
  }

}
