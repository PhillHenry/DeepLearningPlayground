package uk.co.odinconsultants.datatest.spark_mllib.ann

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import uk.co.odinconsultants.htesting.hdfs.HdfsForTesting._
import uk.co.odinconsultants.htesting.spark.SparkForTesting
import uk.co.odinconsultants.htesting.spark.SparkForTesting._

object NeuralNetMain {

  import SparkForTesting.session.implicits._

  def main(args: Array[String]): Unit = {
    new MultilayerPerceptronClassifier()
  }

}
