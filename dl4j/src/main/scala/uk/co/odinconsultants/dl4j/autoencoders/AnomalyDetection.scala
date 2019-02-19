package uk.co.odinconsultants.dl4j.autoencoders

import java.io.FileWriter

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.layers.variational.{VariationalAutoencoder => VAE}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.{CpuBackendNd4jPurger, Nd4j}
import org.nd4j.linalg.learning.config
import uk.co.odinconsultants.data.ClusteredEventsData
import uk.co.odinconsultants.dl4j.MultiDimension._

import scala.util.Try

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
        CpuBackendNd4jPurger.purge()
        val net                   = model(data.timeSeriesSize, activation, i.toLong)
        val (trainIter, testIter) = trainTestData(data)

        net.pretrain(trainIter, nEpochs) // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

        val vae       = net.getLayer(0).asInstanceOf[VAE]
        val outliers  = testNetwork(vae, trainIter, testIter)
        results      += activation -> (results(activation) :+ outliers.length)
        println(s"[${new java.util.Date()}]: Sample #$i: Number of outliers: ${outliers.length}")
        net.clear()
//        Nd4j.getMemoryManager.purgeCaches() // UnsupportedOperationException
      }
      val activationStats = statsFor(activation, results(activation))
      printResult(activationStats)
    }

    println("===============================")

    def statsFor(a: Activation, xs: Seq[Int]): (Activation, Double, Double) = {
      val ns = xs.map(_.toDouble).filterNot(_ == 0d) // the DL4J guys warned me against purging (they say a fix is coming soon). In the meantime I sometimes see 0s. Ignore.
      val mu = mean(ns)
      val sd = stdDev(ns)
      (a, mu, sd)
    }

    def printResult(result: (Activation, Double, Double)): Unit = {
      val (a, mu, sd) = result
      println(s"$a: mu = $mu sd = $sd")
    }

    val stats = results.toList.sortBy(_._1.toString).map { case (a, xs) =>
      statsFor(a, xs)
    }

    stats foreach printResult

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
      Try { features.close() }
      Try { reconstructionErrorEachExample.close() }
      Try { labels.close() }
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
    * Out of 25:
    * CUBE:         mu = 12.777777777777779 sd = 0.9718253158075499
    * ELU:          mu = 14.6               sd = 0.6992058987801011
    * HARDSIGMOID:  mu = 15.0               sd = 0.0
    * HARDTANH:     mu = 14.5               sd = 0.5270462766947299
    * IDENTITY:     mu = 13.9               sd = 0.875595035770913
    *
    * LEAKYRELU:    mu = 14.7               sd = 0.48304589153964794
    * RATIONALTANH: mu = 14.3               sd = 0.48304589153964794
    * RELU:         mu = 14.7               sd = 0.48304589153964794
    * RELU6:        mu = 14.7               sd = 0.48304589153964794
    * RRELU:        mu = 14.5               sd = 0.5270462766947299
    *
    * SIGMOID:      mu = 15.0               sd = 0.0
    * SOFTMAX:      mu = 15.0               sd = 0.0
    * SOFTPLUS:     mu = 14.7               sd = 0.6749485577105528
    * SOFTSIGN:     mu = 14.9               sd = 0.31622776601683794
    * TANH:         mu = 14.5               sd = 0.5270462766947299
    *
    * RECTIFIEDTANH: mu = 14.9              sd = 0.31622776601683794
    * SELU:         mu = 14.5               sd = 0.5270462766947299
    * SWISH:        mu = 15.0               sd = 0.4714045207910317
    * THRESHOLDEDRELU: mu = 14.8            sd = 0.7888106377466155
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
        .activation(activation)
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
    net.addListeners(new ScoreIterationListener(1000))

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
