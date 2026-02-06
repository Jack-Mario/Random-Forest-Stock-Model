import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame, period: int = 60, alfa: float = 0.1) -> pd.DataFrame:
    """Add technical indicators used by the project.

    Keeps original OHLCV columns and appends indicators.
    """
    df = df.copy()
    df['Increase'] = np.where(df['Close'].shift(-period) > df['Close'], 1, -1)
    df['Smooth'] = df['Close'].ewm(alpha=alfa, adjust=False).mean()
    df['Delta_Smooth'] = df['Smooth'].diff()

    # RSI
    df['RSI_gain'] = df['Delta_Smooth'].clip(lower=0)
    df['RSI_loss'] = (-df['Delta_Smooth']).clip(lower=0)
    df['RSI_gain_mean'] = df['RSI_gain'].rolling(14).mean()
    df['RSI_loss_mean'] = df['RSI_loss'].rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + df['RSI_gain_mean'].div(df['RSI_loss_mean']))

    # Stochastic Oscillator
    df['L(14)'] = df['Low'].rolling(14).min()
    df['H(14)'] = df['High'].rolling(14).max()
    df['K(14)'] = 100 * ((df['Smooth'] - df['L(14)']).div(df['H(14)'] - df['L(14)']))

    # Williams
    df['R(14)'] = -100 * ((df['H(14)'] - df['Smooth']).div(df['H(14)'] - df['L(14)']))

    # EMA / MACD
    df['Smooth12'] = df['Smooth'].ewm(span=12, adjust=False).mean()
    df['Smooth26'] = df['Smooth'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['Smooth12'] - df['Smooth26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # PROC
    df['PROC'] = (df['Smooth'] - df['Smooth'].shift(10)) / df['Smooth'].shift(10)

    # OBV
    df['OBV'] = (
        np.sign(df['Smooth'].diff())
        .fillna(0)
        .mul(df['Volume'])
        .cumsum()
    )

    return df
