package uk.co.odinconsultants.dl4j.autoencoders

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, VariationalAutoencoder}
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

      override def N: Int = 600

      override def timeSeriesSize: Int = 50
    }
    import data._
    val (train, test) = trainTest(Seq(xs), 0.9)
    val nClasses      = 2
    val nIn           = timeSeriesSize
    val net           = model(nIn)
    val nEpochs       = 5

    val jTrain        = to2DDataset(train, nClasses, timeSeriesSize)
    val trainIter     = new ListDataSetIterator(jTrain.batchBy(1), 10)

    val testDataSets  = test.map(x => to2DDataset(Seq(x), nClasses, data.timeSeriesSize)).toList
    val jTestDataSets = testDataSets.asJava
    val testIter      = new ListDataSetIterator(jTestDataSets)

    net.pretrain(trainIter, nEpochs) // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

    testDataSets.foreach { ds =>
      println(ds.getFeatures.length()) // "50"
//      println(m.score(ds)) // Exception: Cannot calculate score if final layer is not an instance of IOutputLayer. Final layer is of type: o.d.n.l.v.VariationalAutoencoder
    }

    val vae = net.getLayer(0).asInstanceOf[VariationalAutoencoder]

    while (testIter.hasNext) {
      val ds = testIter.next
      val features = ds.getFeatures
      val labels = Nd4j.argMax(ds.getLabels, 1)
      //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
      val nRows = features.rows
      //Calculate the log probability for reconstructions as per An & Cho
      //Higher is better, lower is worse
      //Shape: [minibatchSize, 1]
      var j = 0
      while (j < nRows) {
        val example = features.getRow(j)
        val reconstructionErrorEachExample = vae.getOutputDistribution.negLogProbability(features, example, true)
        val label = labels.getDouble(j: Long).toInt
//        val score = reconstructionErrorEachExample.getDouble(j)
        j += 1;
      }
    }


    net
  }

  /**
    * Taken from Alex Black's VariationalAutoEncoderExample in DeepLearning4J examples.
    */
  def model(nIn: Int): MultiLayerNetwork = {
    val rngSeed = 12345
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new RmsProp(1e-2))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-4)
      .list()
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(Activation.LEAKYRELU)
        .encoderLayerSizes(nIn/2)
        .decoderLayerSizes(nIn/2)
        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
        .nIn(nIn)
        .nOut(nIn)
        .build())
//      .pretrain(true) // doesn't affect training any more. Use org.deeplearning4j.nn.multilayer.MultiLayerNetwork#pretrain(DataSetIterator) when training for layerwise pretraining.
//      .backprop(false) // doesn't affect training any more. Use org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator) when training for backprop.
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(10))
    net
  }


}
