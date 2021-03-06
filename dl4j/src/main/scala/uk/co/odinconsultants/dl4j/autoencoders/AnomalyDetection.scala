package uk.co.odinconsultants.dl4j.autoencoders

import java.io.FileWriter

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, GaussianReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.layers.variational.{VariationalAutoencoder => VAE}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.factory.{CpuBackendNd4jPurger, Nd4j}
import org.nd4j.linalg.learning.config
import org.nd4j.linalg.learning.config.{AdaDelta, RmsProp}
import uk.co.odinconsultants.data.ClusteredEventsData
import uk.co.odinconsultants.dl4j4s.data.DataSetShaper

import scala.util.Try

object AnomalyDetection {

  def main(args: Array[String]): Unit = {
    println(process())
  }

  def process(): Unit = {

    def statsFor(a: Axis, xs: Seq[Int]): RunStat = {
      val ns = xs.map(_.toDouble).filterNot(_ == 0d) // the DL4J guys warned me against purging (they say a fix is coming soon). In the meantime I sometimes see 0s. Ignore.
      val mu = mean(ns)
      val sd = stdDev(ns)
      (a, mu, sd)
    }

    val bunched2SpreadRatio = 0.0025
    val N                   = 10000
    val timeSeriesSize      = 50
    val nEpochs             = 100
    val nSamples            = 10
    type Axis               = Activation
    val results             = collection.mutable.Map[Axis, Seq[Int]]().withDefault(_ => Seq.empty)
    val l2                  = 1
    val batch               = 64
    println(s"batch: $batch")
    val activation = Activation.SOFTPLUS
//    for (activation <- Activation.values().drop(6)) {
      for (i <- 1 to nSamples) {
//        CpuBackendNd4jPurger.purge()
        val data = new ClusteredEventsData(bunched2SpreadRatio, N, timeSeriesSize, i)
        val net = model(timeSeriesSize, activation, i.toLong, l2)
        val (trainIter, testIter) = trainTestData(data, batch.toInt)

        net.pretrain(trainIter, nEpochs)

        val vae = net.getLayer(0).asInstanceOf[VAE]
        val outliers = testNetwork(vae, trainIter, testIter)
        results += activation -> (results(activation) :+ outliers.length)
        println(s"[${new java.util.Date()}]: Sample #$i: Number of outliers: ${outliers.length}")
//        net.clear()
      }
      val activationStats = statsFor(activation, results(activation))
      printResult(activationStats)

      println("===============================")
//    }

    type RunStat  = (Axis, Double, Double)

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

    val shaper        = new DataSetShaper[Long]
    val jTrain        = shaper.to2DDataset(spread, nClasses, seriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), batchSize)

    val testDataSets  = shaper.to2DDataset(bunched, nClasses, seriesSize)
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
    * SOFTPLUS,       layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTPLUS,     reconstruction = SOFTPLUS     updater = Adam(1e-3):     mu = 19.1 sd = 3.5418137224371984
    * HARDSIGMOID,    layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = HARDSIGMOID,  reconstruction = HARDSIGMOID  updater = Adam(1e-3):     mu = 18.9 sd = 3.178049716414141
    * SOFTMAX,        layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTMAX,      reconstruction = SOFTMAX      updater = Adam(1e-3):     mu = 18.8 sd = 3.645392830531285
    * SIGMOID,        layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SIGMOID,      reconstruction = SIGMOID      updater = Adam(1e-3):     mu = 18.8 sd = 3.2591750830880843
    * RECTIFIEDTANH,  layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-3):     mu = 18.6 sd = 3.4383458555273667
    * SWISH,          layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-3):     mu = 18.6 sd = 3.4383458555273667
    * SWISH,          layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-4):     mu = 18.6 sd = 3.4383458555273667
    * TANH,           layers = [nIn],         l2=1,       batchsize=64,   pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-4):     mu = 18.6 sd = 3.4383458555273667
    * SWISH,          layers = [nIn],         l2=1,       batchsize=128,  pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-4):     mu = 18.6 sd = 3.4383458555273667
    * SWISH,          layers = [nIn],         l2=1e-1,    batchsize=128,  pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-4):     mu = 18.6 sd = 3.4383458555273667
    * TANH,           layers = [nIn],         l2=1e-5,    batchsize=128,  pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-4):     mu = 18.5 sd = 3.4075080500434787
    * SWISH,          layers = [nIn],         l2=1e-3,    batchsize=128,  pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-5):     mu = 18.3 sd = 3.4334951418181574
    * SWISH,          layers = [nIn],         l2=1e-1,    batchsize=128,  pzActivation  = SOFTMAX,      reconstruction = SOFTMAX,     updater = Adam(1e-5):     mu = 18.2 sd = 3.4253953543107016
    */
  def model(nIn: Int, activation: Activation, rngSeed: Long, l2: Double): MultiLayerNetwork = {
    println(s"activation = $activation, l2 = $l2")
    val hiddenLayerSize = nIn / 2

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new config.Adam(1e-4))
      .weightInit(WeightInit.XAVIER)
      .l2(l2)
      .list()
//      .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(nHidden).build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX).nIn(nHidden).nOut(nClasses).lossFunction(new LossNegativeLogLikelihood(Nd4j.create(Array(0.005f, 1f)))).build())
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(activation)
        .encoderLayerSizes(hiddenLayerSize)
        .decoderLayerSizes(hiddenLayerSize)
        .pzxActivationFunction(activation)  //p(z|data) activation function
        .reconstructionDistribution(new GaussianReconstructionDistribution(activation))
        .nIn(nIn)
        .nOut(nIn)
        .build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.addListeners(new ScoreIterationListener(100))
    uiServerListensTo(net)
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
//    net.addListeners(new StatsListener(statsStorage))
    net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100))
    net
  }


}
