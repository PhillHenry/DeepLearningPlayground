package uk.co.odinconsultants.data

import java.time.LocalDateTime

import org.scalatest.{Matchers, WordSpec}

class DateTimeUtilsSpec extends WordSpec with Matchers {

  import DateTimeUtils._

  "Gaps" should {
    val _1Jan2018 = LocalDateTime.of(2018, 1, 1, 0, 0)
    val _1Feb2018 = LocalDateTime.of(2018, 2, 1, 0, 0)
    val _1Mar2018 = LocalDateTime.of(2018, 3, 1, 0, 0)
    val dates     = Seq(_1Jan2018, _1Feb2018, _1Mar2018)
    val gaps      = gapsInSeconds(dates)
    "be one fewer than points" in {
      gaps should have size (dates.size -1)
    }
    "be at least 28 days" in {
      gaps.foreach( s =>
        s should be >= (28 * SECONDS_IN_DAY)
      )
    }
    "be less than 32 days" in {
      gaps.foreach( s =>
        s should be < (32 * SECONDS_IN_DAY)
      )
    }
  }

}
