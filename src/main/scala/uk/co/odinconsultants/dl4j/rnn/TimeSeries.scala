package uk.co.odinconsultants.dl4j.rnn

import java.time.ZoneOffset
import java.util.{List => JList}

import org.datavec.api.records.reader.impl.inmemory.InMemorySequenceRecordReader
import org.datavec.api.writable.{IntWritable, LongWritable, Writable}
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import uk.co.odinconsultants.data.TimeSeriesGenerator._

import scala.collection.JavaConverters._
import scala.util.Random


object TimeSeries {



  def process(): MultiLayerNetwork = {

    def noisyTime(offsetHour: Int): GenerateFn = { time =>
      val rndOffset = (Random.nextGaussian() * 120).toLong
      val noisy     = time.plusMinutes(rndOffset)
      Seq(noisy.plusHours(offsetHour).toEpochSecond(ZoneOffset.of("Z")))
    }

    val start         = DDMMYYYY(1, 1, 2013)
    val end           = DDMMYYYY(1, 1, 2014)
    val nightTimes    = (1 to 300).flatMap(_ => generate(start, end, noisyTime(0))).map(_ -> 1)
    val dayTimes      = (1 to 300).flatMap(_ => generate(start, end, noisyTime(12))).map(_ -> 0)
    val data          = nightTimes ++ dayTimes
    val xs            = Random.shuffle(data)
    val trainSize     = (xs.size * 0.9).toInt
    val train         = xs.take(trainSize)
    val test          = xs.drop(trainSize)
    val nClasses      = 2

    val jTrain = toDatasetIterator(toJLists(train), nClasses)
    val jTest  = toDatasetIterator(toJLists(test), nClasses)

    val m = model(nClasses.toInt)
    val nEpochs = 30
    (1 to nEpochs).foreach { _ =>
      m.fit(jTrain)
    }

    val evaluation: Evaluation = m.evaluate(jTest)

    // print the basic statistics about the trained classifier
    println("Accuracy: "+evaluation.accuracy)
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
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.005, 0.01))
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
      .pretrain(false).backprop(true).build()

    val model = new MultiLayerNetwork(conf)
    model.setListeners(new ScoreIterationListener(20))
    model
  }

  def main(xs: Array[String]): Unit = {
    println(process())
  }

}
