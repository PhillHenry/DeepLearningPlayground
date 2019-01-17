package uk.co.odinconsultants.io

import java.io.File
import java.nio.charset.Charset.defaultCharset

import org.apache.commons.io.FileUtils.writeStringToFile

object FilePersister {

  def persist[K, V](dirName: String, vks: Seq[(Seq[V], K)]): (File, File) = {
    val baseDir      = new File(dirName)
    val featuresDir  = new File(baseDir, "features")
    val labelsDir    = new File(baseDir, "labels")

    val dirs = Array(baseDir, featuresDir, labelsDir)
    dirs.foreach(_.mkdirs())

    vks.zipWithIndex.foreach { case ((vs, k), i) =>
      val labelFile     = new File(labelsDir,   i + ".csv")
      val featuresFile  = new File(featuresDir, i + ".csv")
      writeStringToFile(labelFile, k.toString, defaultCharset)
      writeStringToFile(featuresFile, vs.mkString("\n"), defaultCharset)
    }

    (featuresDir, labelsDir)
  }

}
