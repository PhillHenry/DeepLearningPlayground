package uk.co.odinconsultants.data

trait ClusteredEventsData {

  def ratioRedTo1Blue: Int

  def N: Int

  def timeSeriesSize: Int

  val nBlue: Int = N / (1 + ratioRedTo1Blue)

  val nRed: Int = N - nBlue

  val red = (1 to nRed).map { _ =>
    ???
  }

}
