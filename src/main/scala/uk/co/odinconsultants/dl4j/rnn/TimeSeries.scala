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

    val jTrain = to3DDataset(train, nClasses)
    //val jTest  = toDatasetIterator(test, nClasses)

    val m = model(nClasses.toInt)
    val nEpochs = 10
    (1 to nEpochs).foreach { i =>
      println(s"Epoch $i")
      m.fit(jTrain)
    }
/*
    val evaluation: Evaluation = m.evaluate(jTest)
    println("Accuracy: "+evaluation.accuracy)
    println("Accuracy: "+evaluation.stats())
    println("Precision: "+evaluation.precision)
    println("Recall: "+evaluation.recall)
*/
    m
  }

  type Series2Cat = (Seq[Long], Int)

  def to3DDataset(s2cs: Seq[Series2Cat], nClasses: Int): DataSet = {
    val format   = Nd4j.order
    val features = Nd4j.zeros(s2cs.size, 1, s2cs.head._1.size)
    val labels   = Nd4j.zeros(s2cs.size, nClasses, s2cs.head._1.size)
    s2cs.zipWithIndex.foreach { case ((xs, c), i) =>
      xs.zipWithIndex.foreach { case (x, j) =>
        val indxLabels:   Array[Int] = Array(i, c, j)
        val indxFeatures: Array[Int] = Array(i, 0, j)
        features.putScalar(indxFeatures, x)
        labels.putScalar(indxLabels, 1)
      }
    }
    features.setOrder('f')
    labels.setOrder('f')
    new DataSet(features, labels)
  }

  def toFeatureArray(xs: Seq[Long]): NDArray = new NDArray(Array(xs.map(_.toDouble).toArray))

  def toLabelArray(c: Int, xs: Seq[Long], nClasses: Int): INDArray = { // see BasicRNNExample
    val labels = Nd4j.zeros(1, xs.size, nClasses)
//    new NDArray(Array(Array.fill(size)(c.toDouble)))
    xs.zipWithIndex.map(_._2).foreach { i =>
      val indices: Array[Int] = Array(0, i, c)
      labels.putScalar(indices, 1);
    }
    labels
  }

  def to3DDataset(nClasses: Int): ((Seq[Series2Cat])) => (NDArray, NDArray) = { s2c =>
    val features    = s2c.map(_._1).map(toFeatureArray).asInstanceOf[Seq[INDArray]].asJava
    val labels      = s2c.map { case (xs, c) => toLabelArray(c, xs, nClasses) }.asJava
    val dimensions  = Array(s2c.size, 1, s2c.head._1.length)
    val format      = Nd4j.order
    (new NDArray(features, dimensions, format), new NDArray(labels, dimensions, format))
  }

  def toDatasetIterator(xs: Seq[Series2Cat], nClasses: Int): DataSetIterator = {
    val fn3d      = to3DDataset(nClasses)
    val datasets  = xs.grouped(7).map(fn3d).map { case (f, l) => new DataSet(f, l)}
    new ListDataSetIterator(datasets.toList.asJava)
  }

  def toDataset(xs: Seq[Series2Cat], nClasses: Int): DataSet = {
    val (features, labels) = to3DDataset(nClasses)(xs)
    println(s"features = ${features.shape().mkString(", ")}, labels = ${labels.shape().mkString(",")}")
    new DataSet(features, labels)
  }

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
