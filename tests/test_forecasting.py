import pandas as pd
import unittest

from forecasting import forecast_sentiment, prepare_daily_sentiment


class TestForecasting(unittest.TestCase):
    def test_prepare_daily_sentiment_detects_synthetic_history(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
                "sentiment": [0.2, 0.4, 0.1],
            }
        )

        daily_df, synthetic_history = prepare_daily_sentiment(df)

        self.assertTrue(synthetic_history)
        self.assertEqual(len(daily_df), 30)
        self.assertTrue((daily_df["sentiment"] >= -1.0).all())
        self.assertTrue((daily_df["sentiment"] <= 1.0).all())

    def test_forecast_sentiment_returns_synthetic_flag(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
                "sentiment": [0.2, 0.4, 0.1],
            }
        )

        forecast_df, synthetic_history = forecast_sentiment(df, forecast_days=7)

        self.assertTrue(synthetic_history)
        self.assertEqual(len(forecast_df), 7)
        self.assertIn("predicted_sentiment", forecast_df.columns)


if __name__ == "__main__":
    unittest.main()
