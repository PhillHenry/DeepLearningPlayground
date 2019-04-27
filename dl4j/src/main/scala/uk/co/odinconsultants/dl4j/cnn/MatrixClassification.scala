package uk.co.odinconsultants.dl4j.cnn

import java.util
import java.util.{HashMap, Map}

import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{MapSchedule, ScheduleType}

object MatrixClassification {

  def main(args: Array[String]): Unit = {
    val seed = 1L
    val learningRateSchedule: util.Map[java.lang.Integer, java.lang.Double] = new util.HashMap[java.lang.Integer, java.lang.Double]
    learningRateSchedule.put(0, 0.06)
    learningRateSchedule.put(200, 0.05)
    learningRateSchedule.put(600, 0.028)
    learningRateSchedule.put(800, 0.0060)
    learningRateSchedule.put(1000, 0.001)
    val updater = new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule))
    val channels = 1
    val outputNum = 2
    val height = 100
    val width = 100

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
    net.setListeners(new ScoreIterationListener(10))
  }

}
