package uk.co.odinconsultants.dl4j.autoencoders

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer, variational}
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, GaussianReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config
import org.nd4j.linalg.learning.config.{Adam, RmsProp}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood
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
      override def bunched2SpreadRatio: Double = 0.0025

      override def N: Int = 10000

      override def timeSeriesSize: Int = 50
    }
    import data._
//    val (train, test) = trainTest(Seq(xs), 0.9)
    val nClasses      = 2
    val nIn           = timeSeriesSize
    val net           = model(nIn)
    val nEpochs       = 100

    val jTrain        = to2DDataset(spread, nClasses, timeSeriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)

    val testDataSets  = to2DDataset(bunched, nClasses, timeSeriesSize)
    val testIter      = new ListDataSetIterator(testDataSets.batchBy(1), 10)

    val normalizer = new NormalizerStandardize
    normalizer.fit(trainIter)

    trainIter.reset()

    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    net.pretrain(trainIter, nEpochs) // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

    val vae = net.getLayer(0).asInstanceOf[org.deeplearning4j.nn.layers.variational.VariationalAutoencoder]

    trainIter.reset()
    println("Training:")
    val trainStats = stats(reconstructionCostsOf(trainIter, vae))(SPREAD)
    println("Testing:")
    val testStats: Results = stats(reconstructionCostsOf(testIter, vae))(BUNCHED)
    val validOutliers = testStats.costs.filter(x => x < trainStats.min || x > trainStats.max)
    println(s"${validOutliers.length} of ${testStats.costs.length} are outliers (${validOutliers.length.toDouble * 100 / testStats.costs.length} %)")

    net
  }

  case class Results(mu: Double, sd: Double, min: Double, max: Double, n: Int, costs: List[Double])

  def stats(results: Map[Int, List[Double]]): Map[Int, Results] = {
    results.map { case (l, xs) =>
      val x = Results(mean(xs), stdDev(xs), xs.min, xs.max, xs.length, xs)
      import x._
      println(s"$l: mean = $mu, std dev = $sd, min = $min, max = $max, size = $n ${if (n < 10) "[" + xs.sorted.mkString(", ") + "]" else ""}")
      l -> x
    }
  }

  def reconstructionCostsOf(iter: ListDataSetIterator[DataSet], vae: org.deeplearning4j.nn.layers.variational.VariationalAutoencoder): Map[Int, List[Double]] = {
    val results = collection.mutable.Map[Int, List[Double]]().withDefault(_ => List())
    while (iter.hasNext) {
      val ds        = iter.next
      val features  = ds.getFeatures
      val labels    = Nd4j.argMax(ds.getLabels, 1)  //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
      val nRows     = features.rows
      //Calculate the log probability for reconstructions as per An & Cho
      //Higher is better, lower is worse
      //Shape: [minibatchSize, 1]
      var j = 0
      val reconstructionErrorEachExample = vae.reconstructionLogProbability(features, 5)
      while (j < nRows) {
        val label = labels.getDouble(j: Long).toInt
        val score = reconstructionErrorEachExample.getDouble(j: Long)
        results += label -> (results(label.toInt) :+ score)
        j += 1
      }
    }
    results.toMap
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
    val nHidden         = 20
    val nClasses        = 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new config.Adam(1e-5))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-5)
      .list()
//      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(nHidden).build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX).nIn(nHidden).nOut(nClasses).lossFunction(new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))).build())
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(Activation.THRESHOLDEDRELU) // CUBE 60%; HARDSIGMOID 68%; HARDTANH 64%; LEAKYRELU 76%; RATIONALTANH 64%; RECTIFIEDTANH 84%, 68%; RELU 64%; RELU6 56%; RRELU 56%; SELU 68%; SIGMOID 76%; SOFTMAX 72%; SOFTPLUS 68%; SOFTSIGN 80%; SWISH 60%; TANH 72%; THRESHOLDEDRELU 72%
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
