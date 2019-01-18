package uk.co.odinconsultants.dl4j.autoencoders

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.RmsProp

object AnomalyDetection {

  // Note use ".pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training"

  /**
    * Taken from Alex Black's VariationalAutoEncoderExample in DeepLearning4J examples.
    */
  def autoEncoder = {
    val rngSeed = 12345
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .updater(new RmsProp(1e-2))
      .weightInit(WeightInit.XAVIER)
      .l2(1e-4)
      .list()
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(Activation.LEAKYRELU)
        .encoderLayerSizes(256, 256)        //2 encoder layers, each of size 256
        .decoderLayerSizes(256, 256)        //2 decoder layers, each of size 256
        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
        .nIn(28 * 28)                       //Input size: 28x28
        .nOut(2)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
        .build())
//      .pretrain(true) // doesn't affect training any more. Use org.deeplearning4j.nn.multilayer.MultiLayerNetwork#pretrain(DataSetIterator) when training for layerwise pretraining.
//      .backprop(false) // doesn't affect training any more. Use org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator) when training for backprop.
      .build()

    val net = new MultiLayerNetwork(conf)
    net.setListeners(new ScoreIterationListener(1))
  }


}
