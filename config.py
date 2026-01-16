import os
from pathlib import Path
import json
import logging

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

USER_SETTINGS_FILE = CONFIG_DIR / "user_settings.json"

STOCK_LIST = ['000001', '000002', '000063', '000333', '000651', '000858', '002415', '002594', '600000', '600036', '600519', '600887', '601318', '601398', '601857', '601939', '603259']

DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2025-12-31'

TRAIN_TEST_SPLIT = 0.8
LOOKBACK_DAYS = 20
PREDICTION_DAYS = 5

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

BACKTEST_PARAMS = {
    'initial_cash': 100000,
    'commission': 0.001,
    'slippage': 0.001
}

from storage import storage

def load_user_settings():
    return storage.load_json(USER_SETTINGS_FILE, "user_settings", {})

def save_user_settings(settings):
    storage.save_json(USER_SETTINGS_FILE, "user_settings", settings)
