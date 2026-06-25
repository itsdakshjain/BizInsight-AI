import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def prepare_daily_sentiment(df):
    """
    Convert review-level data into daily average sentiment.
    """

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    daily = data.groupby(data["date"].dt.date)["sentiment"].mean().reset_index()

    daily.columns = ["date", "sentiment"]

    synthetic_history = False

    # Demo data generation for testing
    # If all reviews were uploaded on the same day,
    # create synthetic 30-day history.
    if len(daily) == 1:

        synthetic_history = True
        last_sentiment = daily["sentiment"].iloc[0]

        demo_dates = pd.date_range(end=daily["date"].iloc[0], periods=30)

        demo_sentiments = [
            np.clip(last_sentiment + np.random.uniform(-0.15, 0.15), -1, 1)
            for _ in range(30)
        ]

        daily = pd.DataFrame({"date": demo_dates, "sentiment": demo_sentiments})

    daily["date"] = pd.to_datetime(daily["date"])

    return daily, synthetic_history


def create_lag_features(daily_df, n_lags=7):
    """
    Create lag features.
    """

    df = daily_df.copy()

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["sentiment"].shift(lag)

    df = df.dropna()

    return df


def train_model(feature_df, n_lags=7):
    """
    Train Random Forest model.
    """

    X = feature_df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = feature_df["sentiment"]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)

    model.fit(X, y)

    return model


def forecast_sentiment(df, forecast_days=7):
    """
    Generate future sentiment predictions.
    """

    daily_df, synthetic_history = prepare_daily_sentiment(df)

    n_lags = 7

    if len(daily_df) < n_lags + 1:
        raise ValueError(f"At least {n_lags + 1} days of sentiment data are required.")

    feature_df = create_lag_features(daily_df, n_lags=n_lags)

    model = train_model(feature_df, n_lags=n_lags)

    history = daily_df["sentiment"].tolist()

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    future_predictions = []

    for _ in range(forecast_days):

        latest_values = history[-n_lags:]
        X_future = pd.DataFrame([latest_values], columns=feature_cols)

        prediction = model.predict(X_future)[0]
        prediction = float(np.clip(prediction, -1.0, 1.0))

        future_predictions.append(prediction)
        history.append(prediction)

    future_dates = pd.date_range(
        start=daily_df["date"].max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )

    forecast_df = pd.DataFrame(
        {"date": future_dates, "predicted_sentiment": future_predictions}
    )

    return forecast_df, synthetic_history
