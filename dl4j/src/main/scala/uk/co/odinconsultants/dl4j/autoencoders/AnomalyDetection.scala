package uk.co.odinconsultants.dl4j.autoencoders

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, GaussianReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.primitives.Pair
import uk.co.odinconsultants.data.ClusteredEventsData
import uk.co.odinconsultants.data.SamplingFunctions.trainTest
import uk.co.odinconsultants.dl4j.MultiDimension._

import scala.collection.JavaConverters._

object AnomalyDetection {

  def main(args: Array[String]): Unit = {
    println(process())
  }

  def process(): MultiLayerNetwork = {

    val data = new ClusteredEventsData {
      override def bunched2SpreadRatio: Double = 0.01

      override def N: Int = 4000

      override def timeSeriesSize: Int = 50
    }
    import data._
    val (train, test) = trainTest(Seq(xs), 0.9)
    val nClasses      = 2
    val nIn           = timeSeriesSize
    val net           = model(nIn)
    val nEpochs       = 48

    val jTrain        = to2DDataset(train, nClasses, timeSeriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)

    val testDataSets  = to2DDataset(test, nClasses, timeSeriesSize)
    val testIter      = new ListDataSetIterator(testDataSets.batchBy(1), 10)

    net.pretrain(trainIter, nEpochs) // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

    val vae = net.getLayer(0).asInstanceOf[org.deeplearning4j.nn.layers.variational.VariationalAutoencoder]

    val results = collection.mutable.Map[Int, List[Double]]().withDefault(_ => List())
    while (testIter.hasNext) {
      val ds        = testIter.next
      val features  = ds.getFeatures
      val labels    = Nd4j.argMax(ds.getLabels, 1)  //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
      val nRows     = features.rows
      //Calculate the log probability for reconstructions as per An & Cho
      //Higher is better, lower is worse
      //Shape: [minibatchSize, 1]
      var j = 0
      val reconstructionErrorEachExample = vae.reconstructionLogProbability(features, 5)
      while (j < nRows) {
        val example = features.getRow(j)
        val label = labels.getDouble(j: Long).toInt
        val score = reconstructionErrorEachExample.getDouble(j: Long)
        val min = example.toDoubleVector.min
        val max = example.toDoubleVector.max
        val diff = ((max - min) / (3600 * 24)).toInt
        results += label -> (results(label.toInt) :+ score)
        println(s"row #$j, label = $label: score = $score, diff = $diff") //, features = $features, example = $example")
        j += 1;
      }
    }

    results.foreach { case (l, xs) =>
      println(s"$l: mean = ${mean(xs)}, std dev = ${stdDev(xs)}, min = ${xs.min}, max = ${xs.max}")
    }

    net
  }

  def mean(xs: List[Double]): Double = xs.sum / xs.length

  def stdDev(xs: List[Double]): Double = {
    val mu = mean(xs)
    math.pow(xs.map(x => math.pow(x - mu, 2)).sum / (xs.length - 1), 0.5)
  }

  /**
    * Taken from Alex Black's VariationalAutoEncoderExample in DeepLearning4J examples.
    */
  def model(nIn: Int): MultiLayerNetwork = {
    val rngSeed         = 12345
    val hiddenLayerSize = nIn / 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new RmsProp(1e-5))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-5)
      .list()
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(Activation.LEAKYRELU)
        .encoderLayerSizes(hiddenLayerSize)
        .decoderLayerSizes(hiddenLayerSize)
        .pzxActivationFunction(Activation.SOFTMAX)  //p(z|data) activation function
        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
        .nIn(nIn)
        .nOut(nIn)
        .build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(100))

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
    net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100))


    net
  }


}
