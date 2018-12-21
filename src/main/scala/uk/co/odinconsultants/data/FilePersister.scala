package uk.co.odinconsultants.data

import java.io.File

object FilePersister {

  def persist[K, V](dirName: String, vks: Seq[(Seq[V], K)]): Unit = {
    val base = new File(dirName)

  }

}
