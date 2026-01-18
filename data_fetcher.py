import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm
import ta
from typing import List, Dict, Any, Optional, Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def fetch_fundamental_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        Fetch fundamental data (PE, PB, Market Cap, etc.)
        Using stock_zh_a_spot_em or similar interface
        """
        try:
            # stock_zh_a_spot_em returns real-time data for all stocks
            # Ideally we want historical fundamental data, but akshare support is limited for free historical PE/PB
            # For simplicity in this demo, we might fetch current spot data and broadcast it (NOT IDEAL for backtest)
            # OR we try to find a historical interface.
            
            # Actually, stock_zh_index_daily_em for index data
            # For individual stock fundamentals history: stock_a_indicator_lg (legu)
            
            stock_code_6 = stock_code.zfill(6)
            
            # Using legu indicator which provides PE/PB history
            # Note: This might be slow and unstable
            df = ak.stock_a_indicator_lg(symbol=stock_code_6)
            
            if df.empty:
                return None
                
            df['date'] = pd.to_datetime(df['trade_date'])
            df = df.rename(columns={
                'pe': 'pe_ratio',
                'pb': 'pb_ratio',
                'ps': 'ps_ratio',
                'total_mv': 'total_market_cap'
            })
            # Select relevant cols
            cols = ['date', 'pe_ratio', 'pb_ratio', 'total_market_cap']
            # Filter cols that exist
            cols = [c for c in cols if c in df.columns]
            
            return df[cols].sort_values('date')
            
        except Exception as e:
            logger.warning(f"Error fetching fundamental data for {stock_code}: {e}")
            return None

    def fetch_index_data(self, start_date: str, end_date: str, index_code: str = "000001") -> Optional[pd.DataFrame]:
        """
        Fetch market index data (e.g. ShangHai Index)
        """
        try:
            df = ak.stock_zh_index_daily_em(symbol="sh" + index_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'close': 'index_close'})
            
            # Calculate index return
            df['market_return'] = df['index_close'].pct_change()
            
            return df[['date', 'market_return']]
        except Exception as e:
            logger.warning(f"Error fetching index data: {e}")
            return None

    def fetch_multiple_stocks(self, stock_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        all_data = []
        
        # 1. Fetch index data first (common for all)
        index_df = self.fetch_index_data(start_date, end_date)
        
        for stock_code in tqdm(stock_codes, desc="Fetching stock data"):
            # A. Basic Price Data
            df = self.fetch_stock_data(stock_code, start_date, end_date)
            
            if df is not None:
                # B. Fundamental Data (Merge)
                # Try to fetch fundamental data
                try:
                    fun_df = self.fetch_fundamental_data(stock_code)
                    if fun_df is not None and not fun_df.empty:
                        df = pd.merge(df, fun_df, on='date', how='left')
                        # Forward fill fundamental data (it doesn't change daily usually)
                        df[['pe_ratio', 'pb_ratio', 'total_market_cap']] = df[['pe_ratio', 'pb_ratio', 'total_market_cap']].ffill()
                except Exception as e:
                    logger.warning(f"Skipping fundamentals for {stock_code}: {e}")

                # C. Index Data (Merge)
                if index_df is not None:
                    df = pd.merge(df, index_df, on='date', how='left')
                
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.warning(f"File {filepath} not found")
            return None


class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Avoid SettingWithCopyWarning
        df = df.copy()
        
        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Calculating technical indicators"):
            # Ensure we are working on a copy
            group = group.copy().sort_values('date').reset_index(drop=True)
            
            # Using ta library which handles Series correctly
            close = group['close']
            high = group['high']
            low = group['low']
            volume = group['volume']
            
            group['ma5'] = ta.trend.sma_indicator(close, window=5)
            group['ma10'] = ta.trend.sma_indicator(close, window=10)
            group['ma20'] = ta.trend.sma_indicator(close, window=20)
            group['ma60'] = ta.trend.sma_indicator(close, window=60)
            
            group['ema12'] = ta.trend.ema_indicator(close, window=12)
            group['ema26'] = ta.trend.ema_indicator(close, window=26)
            
            macd = ta.trend.MACD(close)
            group['macd'] = macd.macd()
            group['macd_signal'] = macd.macd_signal()
            group['macd_diff'] = macd.macd_diff()
            
            group['rsi'] = ta.momentum.rsi(close, window=14)
            
            bb = ta.volatility.BollingerBands(close)
            group['bb_high'] = bb.bollinger_hband()
            group['bb_mid'] = bb.bollinger_mavg()
            group['bb_low'] = bb.bollinger_lband()
            group['bb_width'] = group['bb_high'] - group['bb_low']
            group['bollinger_width'] = group['bb_width'] # Alias for consistency
            
            group['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
            
            group['obv'] = ta.volume.on_balance_volume(close, volume)
            
            stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
            group['stoch_k'] = stoch_rsi.stochrsi_k()
            group['stoch_d'] = stoch_rsi.stochrsi_d()
            
            group['momentum'] = ta.momentum.roc(close, window=10)
            
            group['volume_ma5'] = volume.rolling(window=5).mean()
            group['volume_ratio'] = volume / group['volume_ma5']
            
            group['price_position'] = (close - low) / (high - low + 1e-8)
            
            processed_groups.append(group)
        
        if not processed_groups:
            return df
            
        combined_df = pd.concat(processed_groups, ignore_index=True)
        
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.ffill().bfill()
        
        return combined_df

    def add_return_features(self, df: pd.DataFrame, lookback_days: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        df = df.copy()
        
        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Calculating return features"):
            group = group.copy().sort_values('date').reset_index(drop=True)
            close = group['close']
            
            for days in lookback_days:
                group[f'return_{days}d'] = close.pct_change(days)
                group[f'return_{days}d_max'] = close.rolling(days).max().pct_change(days)
                group[f'return_{days}d_min'] = close.rolling(days).min().pct_change(days)
            
            return_1d = group['return_1d']
            group['volatility_5d'] = return_1d.rolling(5).std()
            group['volatility_10d'] = return_1d.rolling(10).std()
            group['volatility_20d'] = return_1d.rolling(20).std()
            
            group['skewness_10d'] = return_1d.rolling(10).skew()
            group['kurtosis_10d'] = return_1d.rolling(10).kurt()
            
            processed_groups.append(group)
            
        if not processed_groups:
            return df

        combined_df = pd.concat(processed_groups, ignore_index=True)
        
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.ffill().bfill()
        
        return combined_df

    def add_target_variable(self, df: pd.DataFrame, prediction_days: int = 5) -> pd.DataFrame:
        df = df.copy()
        
        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Adding target variable"):
            group = group.copy().sort_values('date').reset_index(drop=True)
            
            group['future_return'] = group['close'].shift(-prediction_days) / group['close'] - 1
            group['target'] = (group['future_return'] > 0).astype(int)
            
            processed_groups.append(group)
        
        if not processed_groups:
            return df
            
        combined_df = pd.concat(processed_groups, ignore_index=True)
        combined_df = combined_df.dropna(subset=['target'])
        
        return combined_df

    def prepare_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        if feature_cols is None:
            exclude_cols = ['stock_code', 'date', '日期', 'open', 'high', 'low', 'close', 'volume', 'amount', 
                           'amplitude', 'pct_change', 'change', 'turnover', 'future_return', 'target']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].copy() if 'target' in df.columns else None
        
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Optimize filling missing values
        # Group columns by type to vectorize fillna
        numeric_cols = X.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            
        # Fill remaining (if any) with 0
        X = X.fillna(0)
        
        return X, y, feature_cols


class DataPreprocessor:
    def __init__(self):
        self.scaler = None

    def normalize_features(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled

    def split_data(self, df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.sort_values('date')
        split_idx = int(len(df) * split_ratio)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], n_std: int = 3) -> pd.DataFrame:
        df = df.copy()
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
