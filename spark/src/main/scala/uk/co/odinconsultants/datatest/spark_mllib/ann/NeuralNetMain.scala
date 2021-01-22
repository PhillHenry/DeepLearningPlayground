package uk.co.odinconsultants.datatest.spark_mllib.ann

import org.apache.spark.ml.ann.{FeedForwardTopology, FeedForwardTopologyGetter}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import uk.co.odinconsultants.htesting.spark.SparkForTesting
import uk.co.odinconsultants.htesting.spark.SparkForTesting._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object NeuralNetMain {

  import SparkForTesting.session.implicits._

  def main(args: Array[String]): Unit = {
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = session.read.format("libsvm").load(args(0))

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model: MultilayerPerceptronClassificationModel = trainer.fit(train)
    examineModel(layers, model)

    // compute accuracy on the test set
    val result = model.transform(test)

    model.weights
    model.save("/tmp/my_nn")

    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  }

  private def examineModel(layers: Array[Int], model: MultilayerPerceptronClassificationModel) = {
    val topology = FeedForwardTopologyGetter(layers)
    val ffModel   = topology.model(model.weights)
    ffModel.layerModels.foreach(println)
    println()
  }
}
