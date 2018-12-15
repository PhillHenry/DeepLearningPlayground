package uk.co.odinconsultants.dl4j.rnn

import java.time.{LocalDateTime, ZoneOffset}

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import uk.co.odinconsultants.data.TimeSeriesGenerator._

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

    val n             = xs.size.toLong
    val nClasses      = 2L
    val obsPerSample  = xs.head._1.size.toLong

    val input         = Nd4j.zeros(trainSize, obsPerSample)
    val labels        = Nd4j.zeros(trainSize, nClasses)

    train.zipWithIndex.foreach { case ((xs, c), i) =>
      val v = Nd4j.zeros(xs.size)
      xs.zipWithIndex.foreach { case(l, i) => v.putScalar(i.toLong, l.toFloat) }
      input.putRow(i, v)
      if (i == 1) {
        labels.putScalar(Array(1), 1)
        labels.putScalar(Array(0), 0)
      }
      else {
        labels.putScalar(Array(1), 1)
        labels.putScalar(Array(0), 0)
      }
    }

    val ds            = new DataSet(input, labels)

    val m = model(nClasses.toInt)
    m.fit(ds)
    m
  }

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
