package uk.co.odinconsultants.dl4j.rnn

import java.time.ZoneOffset
import java.util.{List => JList}

import org.datavec.api.records.reader.impl.inmemory.InMemorySequenceRecordReader
import org.datavec.api.writable.{IntWritable, LongWritable, Writable}
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
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

    val jTrain = toDatasetIterator(toJLists(train), nClasses)
    val jTest  = toDatasetIterator(toJLists(test), nClasses)

    val m = model(nClasses.toInt)
    val nEpochs = 10
    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(jTrain)
    }

    val evaluation: Evaluation = m.evaluate(jTest)

    // print the basic statistics about the trained classifier
    println("Accuracy: "+evaluation.accuracy)
    println("Accuracy: "+evaluation.stats())
    println("Precision: "+evaluation.precision)
    println("Recall: "+evaluation.recall)

    m
  }

  type JWritables = JList[JList[JList[Writable]]]

  def toDatasetIterator(jTrain: JWritables, nClasses: Int): SequenceRecordReaderDataSetIterator = {
    val batchSize = 1
    val trainRR   = new InMemorySequenceRecordReader(jTrain)
    val trainIter = new SequenceRecordReaderDataSetIterator(trainRR, batchSize, nClasses, 1)
    trainIter
  }

  def toJLists(xs: Seq[(Seq[Long], Int)]): JWritables = xs.map { case (xs, c) =>
    xs.map { x =>
      val features = new java.util.ArrayList[LongWritable].asInstanceOf[JList[Writable]]
      features.add(new LongWritable(x))
      features.add(new IntWritable(c))
      features
    }.toList.asJava
  }.asJava

  /**
    * Stolen from https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
    */
  def model(numLabelClasses: Int): MultiLayerNetwork = {
    val tbpttLength = 50
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.05, 1))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(100).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(100).nOut(numLabelClasses).build())
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
