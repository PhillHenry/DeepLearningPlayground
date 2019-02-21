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

    val nEpochs     = 100
    val nSamples    = 5
    type Axis       = Double
    val results     = collection.mutable.Map[Axis, Seq[Int]]().withDefault(_ => Seq.empty)
    val activation  = Activation.SWISH
    val l2          = 1
    for (x <- 6 to 9) {
      val batch = math.pow(2, x)
      println(s"batch: $batch")
      for (i <- 1 to nSamples) {
        CpuBackendNd4jPurger.purge()
        val net                   = model(data.timeSeriesSize, activation, i.toLong, l2)
        val (trainIter, testIter) = trainTestData(data, batch.toInt)

        net.pretrain(trainIter, nEpochs)

        val vae       = net.getLayer(0).asInstanceOf[VAE]
        val outliers  = testNetwork(vae, trainIter, testIter)
        results      += batch -> (results(l2) :+ outliers.length)
        println(s"[${new java.util.Date()}]: Sample #$i: Number of outliers: ${outliers.length}")
        net.clear()
      }
      val activationStats = statsFor(l2, results(l2))
      printResult(activationStats)
    }

    println("===============================")

    type RunStat  = (Axis, Double, Double)

    def statsFor(a: Axis, xs: Seq[Int]): RunStat = {
      val ns = xs.map(_.toDouble).filterNot(_ == 0d) // the DL4J guys warned me against purging (they say a fix is coming soon). In the meantime I sometimes see 0s. Ignore.
      val mu = mean(ns)
      val sd = stdDev(ns)
      (a, mu, sd)
    }

    def printResult(result: RunStat): Unit = {
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

  def trainTestData(data: ClusteredEventsData, batchSize: Int): (Data, Data) = {
    import data._
    //    val (train, test) = trainTest(Seq(xs), 0.9)
    val nClasses      = 2

    val jTrain        = to2DDataset(spread, nClasses, timeSeriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), batchSize)

    val testDataSets  = to2DDataset(bunched, nClasses, timeSeriesSize)
    val testIter      = new ListDataSetIterator(testDataSets.batchBy(1), batchSize)

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
    * CV with different batch values
    * 512.0: mu = NaN sd = 0.0
    * 256.0: mu = 6.0 sd = NaN
    * 128.0: mu = 13.0 sd = NaN
    * 64.0: mu = 17.0 sd = NaN
    * 16.0: mu = 16.0 sd = NaN
    * 32.0: mu = 16.0 sd = NaN
    * 4.0: mu = 17.0 sd = NaN
    * 8.0: mu = 17.0 sd = NaN
    */
  def model(nIn: Int, activation: Activation, rngSeed: Long, l2: Double): MultiLayerNetwork = {
    val hiddenLayerSize = nIn / 2
    val hiddenLayerSize2 = hiddenLayerSize / 2
    val nHidden         = 20
    val nClasses        = 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new config.Adam(1e-5))
      .weightInit(WeightInit.XAVIER)
      .l2(l2)
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
