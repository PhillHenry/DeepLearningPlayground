package uk.co.odinconsultants.dl4j.autoencoders

import java.io.{FileOutputStream, FileWriter}

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer, variational}
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, GaussianReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.layers.variational.{VariationalAutoencoder => VAE}
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
import scala.io.Source

object AnomalyDetection {

  def main(args: Array[String]): Unit = {
    println(process())
  }


  def process(): Unit = {

    val data = new ClusteredEventsData {
      override def bunched2SpreadRatio: Double = 0.0025

      override def N: Int = 10000

      override def timeSeriesSize: Int = 50
    }

    val nEpochs   = 100
    val nSamples  = 10
    val results   = collection.mutable.Map[Activation, Seq[Int]]().withDefault(_ => Seq.empty)
    for (activation <- Activation.CUBE.getDeclaringClass.getEnumConstants) {
      println(s"Activation: $activation")
      for (i <- 1 to nSamples) {
        val net                   = model(data.timeSeriesSize, activation, i.toLong)
        val (trainIter, testIter) = trainTestData(data)

        net.pretrain(trainIter, nEpochs) // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

        val vae       = net.getLayer(0).asInstanceOf[VAE]
        val outliers  = testNetwork(vae, trainIter, testIter)
        results      += activation -> (results(activation) :+ outliers.length)
        println(s"Number of outliers: ${outliers.length}")
      }
    }

    println("===============================")

    val stats = results.toList.sortBy(_._1.toString).map { case (a, xs) =>
      val ns = xs.map(_.toDouble)
      val mu = mean(ns)
      val sd = stdDev(ns)
      println(s"$a: mu = $mu sd = $sd")
      (a, mu, sd)
    }

    val fos = new FileWriter("/tmp/results.txt")
    fos.write(stats.map { case (a, mu, sd) => s"$a,$mu,$sd" }.mkString("\n"))
    fos.close()
  }

  type Data = ListDataSetIterator[DataSet]

  def trainTestData(data: ClusteredEventsData): (Data, Data) = {
    import data._
    //    val (train, test) = trainTest(Seq(xs), 0.9)
    val nClasses      = 2

    val jTrain        = to2DDataset(spread, nClasses, timeSeriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)

    val testDataSets  = to2DDataset(bunched, nClasses, timeSeriesSize)
    val testIter      = new ListDataSetIterator(testDataSets.batchBy(1), 10)

    val normalizer = new NormalizerStandardize
    normalizer.fit(trainIter)

    trainIter.reset()

    trainIter.setPreProcessor(normalizer)
    testIter.setPreProcessor(normalizer)

    (trainIter, testIter)
  }

  def testNetwork(vae: VAE, trainIter: Data, testIter: Data): Seq[Double] = {
    trainIter.reset()
    println("Training:")
    val trainStats = stats(reconstructionCostsOf(trainIter, vae)).head._2
    println("Testing:")
    val testStats: Results = stats(reconstructionCostsOf(testIter, vae)).head._2
    val validOutliers = testStats.costs.filter(x => x < trainStats.min || x > trainStats.max)
    println(s"${validOutliers.length} of ${testStats.costs.length} are outliers (${validOutliers.length.toDouble * 100 / testStats.costs.length} %)")

    validOutliers
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

  def reconstructionCostsOf(iter: ListDataSetIterator[DataSet], vae: VAE): Map[Int, List[Double]] = {
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

  def mean(xs: Seq[Double]): Double = xs.sum / xs.length

  def stdDev(xs: Seq[Double]): Double = {
    val mu = mean(xs)
    math.pow(xs.map(x => math.pow(x - mu, 2)).sum / (xs.length - 1), 0.5)
  }

  /**
    * Taken from Alex Black's VariationalAutoEncoderExample in DeepLearning4J examples.
    */
  def model(nIn: Int, activation: Activation, rngSeed: Long): MultiLayerNetwork = {
    val hiddenLayerSize = nIn / 2
    val hiddenLayerSize2 = hiddenLayerSize / 2
    val nHidden         = 20
    val nClasses        = 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new config.Adam(1e-5))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-1) // RECTIFIEDTANH/1e-4: 84%, 56%/ 1e-6: 76%, 72%/ 1e-3: 64%, 72%/ 1e-2: 64%, 60%/ 1e-1: 80%, 64%
      .list()
//      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(nHidden).build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX).nIn(nHidden).nOut(nClasses).lossFunction(new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))).build())
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(activation) // CUBE 60%; HARDSIGMOID 68%; HARDTANH 64%; LEAKYRELU 76%; RATIONALTANH 64%; RECTIFIEDTANH 84%, 68%, 76%; RELU 64%; RELU6 56%; RRELU 56%; SELU 68%; SIGMOID 76%; SOFTMAX 72%; SOFTPLUS 68%; SOFTSIGN 80%; SWISH 60%; TANH 72%; THRESHOLDEDRELU 72%
        .encoderLayerSizes(hiddenLayerSize) // RECTIFIEDTANH, hiddenLayerSize2 76%
        .decoderLayerSizes(hiddenLayerSize) // RECTIFIEDTANH, hiddenLayerSize2, 2, 68%
        .pzxActivationFunction(Activation.SOFTMAX)  //p(z|data) activation function
        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
        .nIn(nIn)
        .nOut(nIn)
        .build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(100))

    net.addListeners(new ScoreIterationListener(100))

    net
  }

  def uiServerListensTo(net: MultiLayerNetwork): Unit = {

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
    net.addListeners(new StatsListener(statsStorage))

  }


}
