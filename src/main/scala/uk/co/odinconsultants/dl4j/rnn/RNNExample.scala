package uk.co.odinconsultants.dl4j.rnn

import java.io.File

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.datavec.api.split.NumberedFileInputSplit
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.apache.commons.io.{FileUtils, IOUtils}
import java.nio.charset.Charset
import java.util.Random
import java.net.URL

import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * From https://deeplearning4j.org/tutorials/08-rnns-sequence-classification-of-synthetic-control-data
  * but with bugs fixed
  */
object RNNExample {
  def main(args: Array[String]): Unit = {
    val cache = "/tmp"
    val dataPath = new File(cache, "/uci_synthetic_control/")

    if(!dataPath.exists()) {
      val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
      println("Downloading file...")
      val data = IOUtils.toString(new URL(url), Charset.defaultCharset())
      val lines = data.split("\n")

      var lineCount = 0;
      var index = 0

      val linesList = scala.collection.mutable.ListBuffer.empty[String]
      println("Extracting file...")

      for (line <- lines) {
        val count = new java.lang.Integer(lineCount / 100)
        val newLine: String = line.replaceAll("\\s+", ", " + count.toString() + "\n")
        linesList += newLine + ", " + count.toString()
        lineCount += 1
      }
      scala.util.Random.shuffle(linesList)

      for (line <- linesList) {
        val outPath = new File(dataPath, index + ".csv")
        FileUtils.writeStringToFile(outPath, line, Charset.defaultCharset())
        index += 1
      }
      println("Done.")
    } else {
      println("File already exists.")
    }
    val batchSize = 128
    val numLabelClasses = 6

    // training data
    val trainRR = new CSVSequenceRecordReader(0, ", ")
    trainRR.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 449))
    val trainIter = new SequenceRecordReaderDataSetIterator(trainRR, batchSize, numLabelClasses, 1)

    // testing data
    val testRR = new CSVSequenceRecordReader(0, ", ")
    testRR.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 450, 599))
    val testIter = new SequenceRecordReaderDataSetIterator(testRR, batchSize, numLabelClasses, 1)

    val m = model(numLabelClasses)

    val numEpochs = 1
    (1 to numEpochs).foreach(_ => m.fit(trainIter) )

    val evaluation: Evaluation = m.evaluate(testIter)

    // print the basic statistics about the trained classifier
    println("Accuracy: "+evaluation.accuracy())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())
  }

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
}
