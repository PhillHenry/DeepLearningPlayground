package org.apache.spark.ml.ann

object FeedForwardTopologyGetter {

  /**
   * See MultilayerPerceptronClassifier#train
   */
  def apply(layers: Array[Int]): FeedForwardTopology =
    FeedForwardTopology.multiLayerPerceptron(layers, softmaxOnTop = true)

}
