package uk.co.odinconsultants.dl4j.cnn

import java.util

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{MapSchedule, ScheduleType}
import uk.co.odinconsultants.dl4j.autoencoders.AnomalyDetection
import uk.co.odinconsultants.dl4j4s.data.DataSetShaper

object MatrixClassification {
  def main(args: Array[String]): Unit = {
    val h                     = 100
    val w                     = 100
    val (testIter, trainIter) = testTrain(w, h)
    val m                     = model(h, w)
    val nEpochs               = 5
    (1 to nEpochs).foreach { e =>
      println(s"Epoch # $e")
      m.fit(trainIter)
      trainIter.reset()
    }

    val evaluation: Evaluation = m.evaluate(testIter)
    println("Accuracy: "+evaluation.accuracy)
    println("Stats: "+evaluation.stats())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())
  }

  type Data = ListDataSetIterator[DataSet]

  def testTrain(w: Int, h: Int): (Data, Data) = {
    val raw           = new RawData(w, h)
    import raw._
    val shaper        = new DataSetShaper[Double]
    val nClasses      = 2
    val dsTrain       = shaper.to4DDataset(train, nClasses, w, h)
    val dsTest        = shaper.to4DDataset(test, nClasses, w, h)
    val batchSize     = 32
    val trainIter     = new ListDataSetIterator(dsTrain.batchBy(1), batchSize)
    val testIter      = new ListDataSetIterator(dsTest.batchBy(1), batchSize)
    (testIter, trainIter)
  }

  /**
    * Stolen from MnistClassifier in dl4j-examples
    */
  def model(height: Int, width: Int): MultiLayerNetwork = {
    val seed = 1L
    val learningRateSchedule: util.Map[java.lang.Integer, java.lang.Double] = new util.HashMap[java.lang.Integer, java.lang.Double]
    val x = 100
    learningRateSchedule.put(0, 0.06 / x)
    learningRateSchedule.put(200, 0.05 / x)
    learningRateSchedule.put(600, 0.028 / x)
    learningRateSchedule.put(800, 0.0060 / x)
    learningRateSchedule.put(1000, 0.001 / x)
    val updater = new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule))
    val channels = 1
    val outputNum = 2

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005).updater(updater)
      .weightInit(WeightInit.XAVIER).list
      .layer(new ConvolutionLayer.Builder(5, 5).nIn(channels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build)
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut // nIn need not specified in later layers
        (50).activation(Activation.IDENTITY).build)
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build)
      .layer(
        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nOut(outputNum).activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(height, width, channels))
      .build // InputType.convolutional for normal image


    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()

    AnomalyDetection.uiServerListensTo(net)
    net.setListeners(new ScoreIterationListener(10))
    net
  }

}
