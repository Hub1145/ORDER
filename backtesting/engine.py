"""Backtesting engine for the trade & orderbook pipeline"""

import asyncio
import logging
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
import gzip
import pickle
from dataclasses import dataclass, field
from enum import Enum
import re
import io
from dateutil.parser import isoparse
from scipy.stats import skew, kurtosis # NEW: Import skew and kurtosis

from core.exchange_interface import (
    NormalizedOrderBook, NormalizedTrade, NormalizedTicker, OrderBookLevel
)
from features.extractors.advanced_cryptofeed_extractor import AdvancedCryptofeedExtractor
from features.extractors.advanced_tradebook_extractor import AdvancedTradebookExtractor
from features.extractors.enhanced_technical_extractor import EnhancedTechnicalExtractor
from features.extractors.dex_liquidity_extractor import DEXLiquidityExtractor
from features.extractors.futures_open_interest_funding_extractor import FuturesOpenInterestFundingExtractor
from features.extractors.cross_exchange_discrepancy_extractor import CrossExchangeDiscrepancyExtractor
from detection.system import DetectionSystem
from core.gdrive_utils import GoogleDriveAPI

logger = logging.getLogger(__name__)

# Utility functions for parsing CSV content from bytes
def _read_csv_from_bytes(content: bytes) -> pd.DataFrame:
    """Reads CSV content from bytes, handling gzip compression."""
    try:
        # Try to decompress as gzip first
        with gzip.open(io.BytesIO(content), 'rt') as f:
            return pd.read_csv(f)
    except Exception:
        # If not gzip, try reading directly as plain text
        return pd.read_csv(io.BytesIO(content))

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names and data types of a DataFrame."""
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"timestamp", "time", "ts", "date"}: ren[c] = "timestamp"
        elif cl in {"qty", "quantity", "size", "volume", "amount"}: ren[c] = "volume"
        elif cl in {"px", "prc"}: ren[c] = "price"
    if ren: df = df.rename(columns=ren)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True, errors="coerce")
    
    for k in ("price", "volume"):
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    
    return df


class DataSource(Enum):
    """Supported data sources for backtesting"""
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    JSON = "json"
    DATABASE = "database"
    LIVE_RECORDING = "live_recording"
    LIVE_API = "live_api"
    GDRIVE = "gdrive" # Changed from DRIVE to GDRIVE for clarity


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    exchanges: List[str]
    start_time_str: str = "00:00:00" # HH:MM:SS
    end_time_str: str = "23:59:59" # HH:MM:SS
    data_source: DataSource = DataSource.GDRIVE # Default to GDRIVE
    data_path: str = "./historical_data"
    gdrive_api_key: Optional[str] = None # Google Drive API key
    gdrive_orderbook_root_id: Optional[str] = None # Google Drive folder ID for general orderbook data
    gdrive_orderbook_binance_root_id: Optional[str] = None # Google Drive folder ID for Binance orderbook data
    gdrive_tradebook_root_id: Optional[str] = None # Google Drive folder ID for tradebook data
    data_type: str = "all" # "orderbook", "tradebook", or "all"
    
    # Replay settings
    replay_speed: float = 1.0  # 1.0 = real-time, >1 = faster
    orderbook_depth: int = 50
    trade_history_size: int = 1000
    
    # Feature extraction settings
    enable_features: bool = True
    feature_config: Dict[str, Any] = field(default_factory=dict)
    
    # Anomaly detection settings
    enable_anomaly_detection: bool = True
    detection_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    track_performance: bool = True
    save_results: bool = True
    results_path: str = "./backtest_results"
    output: Dict[str, Any] = field(default_factory=dict) # NEW: Add output configuration


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    
    # Event counts
    total_orderbooks: int = 0
    total_trades: int = 0
    total_anomalies: int = 0
    
    # Performance metrics
    processing_time_ms: List[float] = field(default_factory=list)
    feature_extraction_time_ms: List[float] = field(default_factory=list)
    anomaly_detection_time_ms: List[float] = field(default_factory=list)
    
    # Detected patterns
    anomaly_timeline: List[Dict[str, Any]] = field(default_factory=list)
    feature_statistics: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict) # NEW: Nested dict for exchange -> symbol -> stats
    
    # Custom metrics from strategies
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class HistoricalDataLoader:
    """Loads historical data from various sources"""
    
    def __init__(self, config: BacktestConfig, exchange_manager: Any, data_loader: Any):
        self.config = config
        self.exchange_manager = exchange_manager
        self.data_loader_instance = data_loader # Renamed to avoid conflict with HistoricalDataLoader instance
        self.data_cache = {}
        if self.config.gdrive_api_key:
            self.gdrive_api = GoogleDriveAPI(self.config.gdrive_api_key) # Initialize GDrive API
        else:
            self.gdrive_api = None
            logger.warning("Google Drive API key not provided in config. GDrive data loading will not work.")


    async def load_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load orderbook and trade data for a symbol
        
        Returns:
            Tuple of (orderbook_df, trades_df)
        """
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        if self.config.data_source == DataSource.CSV:
            data = await self._load_csv_data(exchange, symbol)
        elif self.config.data_source == DataSource.PARQUET:
            data = await self._load_parquet_data(exchange, symbol)
        elif self.config.data_source == DataSource.PICKLE:
            data = await self._load_pickle_data(exchange, symbol)
        elif self.config.data_source == DataSource.JSON:
            data = await self._load_json_data(exchange, symbol)
        elif self.config.data_source == DataSource.LIVE_API:
            data = await self._load_live_api_data(exchange, symbol)
        elif self.config.data_source == DataSource.GDRIVE: # NEW
            data = await self._load_gdrive_data(exchange, symbol) # Changed method name
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
            
        self.data_cache[cache_key] = data
        return data
        
    async def _load_csv_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files"""
        base_path = Path(self.config.data_path) / exchange / symbol.replace('/', '_')
        
        orderbook_path = base_path / "orderbooks.csv"
        trades_path = base_path / "trades.csv"
        
        orderbook_df = pd.read_csv(orderbook_path, parse_dates=['timestamp'])
        trades_df = pd.read_csv(trades_path, parse_dates=['timestamp'])
        
        # Filter by date range
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= self.config.start_date) &
            (orderbook_df['timestamp'] <= self.config.end_date)
        ]
        trades_df = trades_df[
            (trades_df['timestamp'] >= self.config.start_date) &
            (trades_df['timestamp'] <= self.config.end_date)
        ]
        
        return orderbook_df, trades_df
        
    async def _load_parquet_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Parquet files"""
        base_path = Path(self.config.data_path) / exchange / symbol.replace('/', '_')
        
        orderbook_df = pd.read_parquet(base_path / "orderbooks.parquet")
        trades_df = pd.read_parquet(base_path / "trades.parquet")
        
        return orderbook_df, trades_df
        
    async def _load_pickle_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Pickle files"""
        base_path = Path(self.config.data_path) / exchange / symbol.replace('/', '_')
        
        with open(base_path / "orderbooks.pkl", 'rb') as f:
            orderbook_df = pickle.load(f)
        with open(base_path / "trades.pkl", 'rb') as f:
            trades_df = pickle.load(f)
            
        return orderbook_df, trades_df
        
    async def _load_json_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from JSON files"""
        base_path = Path(self.config.data_path) / exchange / symbol.replace('/', '_')
        
        # Support compressed JSON
        orderbook_file = base_path / "orderbooks.json.gz"
        if orderbook_file.exists():
            with gzip.open(orderbook_file, 'rt') as f:
                orderbook_data = json.load(f)
        else:
            with open(base_path / "orderbooks.json", 'r') as f:
                orderbook_data = json.load(f)
                
        trades_file = base_path / "trades.json.gz"
        if trades_file.exists():
            with gzip.open(trades_file, 'rt') as f:
                trades_data = json.load(f)
        else:
            with open(base_path / "trades.json", 'r') as f:
                trades_data = json.load(f)
                
        orderbook_df = pd.DataFrame(orderbook_data)
        trades_df = pd.DataFrame(trades_data)
        
        return orderbook_df, trades_df
        
    async def _load_live_api_data(self, exchange_id: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical OHLCV data from live exchange API."""
        logger.info(f"Fetching historical OHLCV for {symbol} from {exchange_id} API...")
        exchange_adapter = self.exchange_manager.get_exchange(exchange_id)
        if not exchange_adapter:
            raise ValueError(f"Exchange {exchange_id} not found in ExchangeManager.")
        
        # Fetch OHLCV data (e.g., 1-minute candles)
        # Default to 1m timeframe, can be made configurable in BacktestConfig if needed
        ohlcv = await exchange_adapter.fetch_ohlcv(
            symbol=symbol,
            timeframe='1m',
            since=int(self.config.start_date.timestamp() * 1000),
            limit=None # Fetch all available data within the range
        )
        
        if not ohlcv:
            logger.warning(f"No OHLCV data found for {symbol} on {exchange_id} for the specified period.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert OHLCV to DataFrame
        ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
        
        # Filter by end date
        ohlcv_df = ohlcv_df[ohlcv_df['timestamp'] <= self.config.end_date]
        
        # For backtesting with OHLCV, we'll return OHLCV data as 'trades' and empty 'orderbook' for now.
        # A more sophisticated approach would involve generating synthetic orderbook events or modifying
        # the backtesting engine to directly consume OHLCV for certain strategies.
        return pd.DataFrame(), ohlcv_df # Return empty orderbook_df and ohlcv_df as trades_df
        
    async def _load_gdrive_data(self, exchange: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from Google Drive using the new gdrive_utils."""
        if not self.gdrive_api:
            raise ValueError("Google Drive API not initialized. Please provide gdrive_api_key in config.")

        logger.info(f"Loading data for {symbol} on {exchange} from Google Drive...")

        orderbook_df = pd.DataFrame()
        trades_df = pd.DataFrame()

        current_date = self.config.start_date.date()
        end_date_limit = self.config.end_date.date()

        # Define time filters
        start_time_obj = datetime.strptime(self.config.start_time_str, "%H:%M:%S").time()
        end_time_obj = datetime.strptime(self.config.end_time_str, "%H:%M:%S").time()

        while current_date <= end_date_limit:
            year = current_date.year
            month = current_date.month
            
            # Orderbook data loading
            if self.config.data_type == "orderbook" or self.config.data_type == "all":
                # Attempt to load from General Orderbook Root
                if self.config.gdrive_orderbook_root_id:
                    ob_symbol_folder_name_general = symbol.replace('/', '').lower()
                    ob_year_folder_name_general = str(year)
                    ob_month_folder_name_general = f"{month:02d}"
                    
                    try:
                        ob_content_general = await self.gdrive_api.get_file_from_drive(
                            self.config.gdrive_orderbook_root_id,
                            [ob_symbol_folder_name_general, ob_year_folder_name_general, ob_month_folder_name_general],
                            f"{current_date.strftime('%Y-%m-%d')}.csv.gz"
                        )
                        if ob_content_general:
                            monthly_ob_df_general = _read_csv_from_bytes(ob_content_general)
                            monthly_ob_df_general = _normalize_dataframe(monthly_ob_df_general)
                            if not monthly_ob_df_general.empty:
                                # Apply time filtering
                                monthly_ob_df_general = monthly_ob_df_general[
                                    (monthly_ob_df_general['timestamp'].dt.time >= start_time_obj) &
                                    (monthly_ob_df_general['timestamp'].dt.time <= end_time_obj)
                                ]
                                orderbook_df = pd.concat([orderbook_df, monthly_ob_df_general], ignore_index=True)
                                logger.info(f"Loaded {len(monthly_ob_df_general)} general orderbook rows for {symbol} on {exchange} for {current_date} (filtered by time).")
                            else:
                                logger.warning(f"General orderbook DataFrame was empty after normalization for {symbol} on {exchange} for {current_date}.")
                    except Exception as e:
                        logger.warning(f"Error loading general orderbook data for {symbol} on {exchange} for {current_date}: {e}")

                # Attempt to load from Binance-specific Orderbook Root (if exchange is binance)
                if exchange.lower() == 'binance' and self.config.gdrive_orderbook_binance_root_id:
                    ob_symbol_folder_name_binance = symbol.replace('/', '').upper()
                    ob_year_month_folder_name_binance = f"{year}_{month:02d}"

                    try:
                        ob_content_binance = await self.gdrive_api.get_file_from_drive(
                            self.config.gdrive_orderbook_binance_root_id,
                            [ob_symbol_folder_name_binance, ob_year_month_folder_name_binance],
                            f"{current_date.strftime('%Y-%m-%d')}.csv.gz"
                        )
                        logger.debug(f"ob_content_binance: {repr(ob_content_binance)}")
                        if ob_content_binance:
                            logger.debug(f"Attempting to read and normalize Binance orderbook content for {symbol} on {exchange} for {current_date}.")
                            monthly_ob_df_binance = _read_csv_from_bytes(ob_content_binance)
                            logger.debug(f"Read {len(monthly_ob_df_binance)} rows from Binance orderbook CSV for {symbol} on {exchange} for {current_date}.")
                            monthly_ob_df_binance = _normalize_dataframe(monthly_ob_df_binance)
                            logger.debug(f"Normalized Binance orderbook DataFrame has {len(monthly_ob_df_binance)} rows for {symbol} on {exchange} for {current_date}.")
                            
                            if not monthly_ob_df_binance.empty:
                                # Apply time filtering
                                monthly_ob_df_binance = monthly_ob_df_binance[
                                    (monthly_ob_df_binance['timestamp'].dt.time >= start_time_obj) &
                                    (monthly_ob_df_binance['timestamp'].dt.time <= end_time_obj)
                                ]
                                orderbook_df_before_concat = len(orderbook_df)
                                orderbook_df = pd.concat([orderbook_df, monthly_ob_df_binance], ignore_index=True)
                                logger.info(f"Loaded {len(monthly_ob_df_binance)} Binance-specific orderbook rows for {symbol} on {exchange} for {current_date} (filtered by time). Total orderbook_df rows after concat: {len(orderbook_df)} (was {orderbook_df_before_concat}).")
                            else:
                                logger.warning(f"Binance-specific orderbook DataFrame was empty after normalization for {symbol} on {exchange} for {current_date}.")
                    except Exception as e:
                        logger.exception(f"Error loading Binance-specific orderbook data for {symbol} on {exchange} for {current_date}.")
            
            # Tradebook data loading (This block is now separate from orderbook loading)
            if self.config.data_type == "tradebook" or self.config.data_type == "all":
                tr_root_id = self.config.gdrive_tradebook_root_id
                tr_symbol_folder_name = symbol.replace('/', '').upper()
                tr_year_month_folder_name = f"{year}_{month:02d}"
                
                if tr_root_id:
                    try:
                        trade_content = await self.gdrive_api.get_file_from_drive(
                            tr_root_id,
                            [tr_symbol_folder_name, tr_year_month_folder_name],
                            f"{current_date.strftime('%Y_%m_%d')}.csv.gz"
                        )
                        if trade_content:
                            monthly_trades_df = _read_csv_from_bytes(trade_content)
                            monthly_trades_df = _normalize_dataframe(monthly_trades_df)
                            # Apply time filtering
                            monthly_trades_df = monthly_trades_df[
                                (monthly_trades_df['timestamp'].dt.time >= start_time_obj) &
                                (monthly_trades_df['timestamp'].dt.time <= end_time_obj)
                            ]
                            trades_df = pd.concat([trades_df, monthly_trades_df], ignore_index=True)
                            logger.info(f"Loaded {len(monthly_trades_df)} tradebook rows for {symbol} on {exchange} for {current_date} (filtered by time).")
                    except Exception as e:
                        logger.warning(f"Error loading tradebook data for {symbol} on {exchange} for {current_date}: {e}")
            
            current_date += timedelta(days=1) # Increment by day for daily files
            
        # Filter by the exact date range after concatenation
        # This part remains the same, as it filters by full date (including time up to midnight)
        orderbook_df = orderbook_df[
            (orderbook_df['timestamp'] >= self.config.start_date) &
            (orderbook_df['timestamp'] <= self.config.end_date)
        ] if not orderbook_df.empty else pd.DataFrame()
        
        trades_df = trades_df[
            (trades_df['timestamp'] >= self.config.start_date) &
            (trades_df['timestamp'] <= self.config.end_date)
        ] if not trades_df.empty else pd.DataFrame()
        
        return orderbook_df, trades_df
        
        
class BacktestingEngine:
    """Main backtesting engine that replays historical data through the pipeline"""
    
    def __init__(self, config: BacktestConfig, exchange_manager: Any, data_loader: Any): # Using Any for now due to circular import
        self.config = config
        self.exchange_manager = exchange_manager
        self.data_loader_instance = data_loader # Renamed to avoid conflict with HistoricalDataLoader instance
        self.data_loader = HistoricalDataLoader(config, exchange_manager, data_loader)
        
        # Initialize feature extractors if enabled
        if config.enable_features:
            self.feature_extractors = self._initialize_feature_extractors()
        else:
            self.feature_extractors = {}
            
        # Initialize anomaly detection if enabled
        if config.enable_anomaly_detection:
            self.detection_system = DetectionSystem(config.detection_config)
        else:
            self.detection_system = None
            
        # Event callbacks
        self.orderbook_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.feature_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []
        
        # Internal state
        self.current_time = config.start_date
        self.results = BacktestResult(config=config, start_time=datetime.utcnow(), end_time=datetime.utcnow())
        self.orderbook_buffers = defaultdict(lambda: deque(maxlen=100))
        self.trade_buffers = defaultdict(lambda: deque(maxlen=config.trade_history_size))
        
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extractors with config"""
        extractors = {}
        fe_config = self.config.feature_config
        
        extractors['cryptofeed'] = AdvancedCryptofeedExtractor(fe_config.get('cryptofeed', {}))
        extractors['tradebook'] = AdvancedTradebookExtractor(fe_config.get('tradebook', {}))
        extractors['technical'] = EnhancedTechnicalExtractor(fe_config.get('technical', {}))
        extractors['dex'] = DEXLiquidityExtractor(fe_config.get('dex', {}))
        extractors['futures'] = FuturesOpenInterestFundingExtractor(fe_config.get('futures', {}))
        extractors['cross_exchange'] = CrossExchangeDiscrepancyExtractor(fe_config.get('cross_exchange', {}))
        
        return extractors
        
    def add_orderbook_callback(self, callback: Callable) -> None:
        """Add callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
        
    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)
        
    def add_feature_callback(self, callback: Callable) -> None:
        """Add callback for extracted features"""
        self.feature_callbacks.append(callback)
        
    def add_anomaly_callback(self, callback: Callable) -> None:
        """Add callback for detected anomalies"""
        self.anomaly_callbacks.append(callback)
        
    async def run(self) -> BacktestResult:
        """Run the backtest"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        logger.debug("Preparing event timeline...")
        all_events = await self._prepare_event_timeline()
        logger.info(f"Event timeline prepared with {len(all_events)} events.") # Changed to info for visibility
        
        # Process events in chronological order
        try:
            last_event_timestamp = None # Initialize to None
            for i, event in enumerate(all_events): # Added enumerate to track progress
                self.current_time = event['timestamp']
                logger.debug(f"Processing event {i+1}/{len(all_events)} at {self.current_time} (Type: {event['type']})") # New log
                
                # Calculate timing
                start_process = datetime.utcnow()
                
                features = {}
                feature_stats = {}
                anomalies_result = {}

                if event['type'] == 'orderbook':
                    await self._process_orderbook_event(event)
                    if self.config.enable_features:
                        features, feature_stats, anomalies_result = await self._extract_and_process_features(event['exchange'], event['symbol'])
                elif event['type'] == 'trade':
                    await self._process_trade_event(event)
                    if self.config.enable_features:
                        features, feature_stats, anomalies_result = await self._extract_and_process_features(event['exchange'], event['symbol'])
                    
                # Track processing time
                process_time = (datetime.utcnow() - start_process).total_seconds() * 1000
                self.results.processing_time_ms.append(process_time)
                
                # Write unified record if in full mode
                if self.config.output.get("mode") == "full":
                    
                    from core.unified_output import write_unified_record # Import here to avoid circular dependency
                    write_unified_record(
                        config=self.config,
                        exchange=event['exchange'],
                        symbol=event['symbol'],
                        features=features,
                        feature_stats=feature_stats,
                        detectors=anomalies_result.get("detectors", {}),
                        meta_stats=anomalies_result.get("meta_statistics", {}),
                        composite=anomalies_result.get("composite", {})
                    )

                # Control replay speed based on actual time difference between historical events
                if self.config.replay_speed < float('inf') and last_event_timestamp is not None:
                    time_diff_real = (event['timestamp'] - last_event_timestamp).total_seconds()
                    simulated_delay = time_diff_real / self.config.replay_speed
                    # Ensure minimum delay to avoid negative sleep or too frequent context switching
                    if simulated_delay > 0.001: # Cap minimum delay
                        await asyncio.sleep(simulated_delay)
                last_event_timestamp = event['timestamp'] # Update last_event_timestamp
            
            logger.info("All events processed. Finalizing backtest results.") # New log
            # Finalize results
            self.results.end_time = datetime.utcnow()
        except Exception as e:
            logger.exception(f"An error occurred during event processing: {e}")
            self.results.end_time = datetime.utcnow() # Set end time even on error
            # Optionally, decide if you want to save partial results here
            logger.warning("Backtest terminated due to an error. Attempting to save partial results.")
        
        if self.config.save_results:
            await self._save_results()
            
        logger.info("Backtest run completed.") # New log
        return self.results
        
    async def _prepare_event_timeline(self) -> List[Dict[str, Any]]:
        """Prepare chronologically sorted timeline of all events"""
        all_events = []
        
        for exchange in self.config.exchanges:
            for symbol in self.config.symbols:
                try:
                    orderbook_df, trades_df = await self.data_loader.load_data(exchange, symbol)
                    
                    if self.config.data_source == DataSource.LIVE_API:
                        # For LIVE_API, trades_df contains OHLCV. Convert to trade events.
                        for _, row in trades_df.iterrows():
                            all_events.append({
                                'timestamp': row['timestamp'],
                                'type': 'trade',
                                'exchange': exchange,
                                'symbol': symbol,
                                'data': {
                                    'price': row['close'], # Use close price as trade price
                                    'volume': row['volume'],
                                    'side': 'buy' if row['open'] <= row['close'] else 'sell' # Simple heuristic for side
                                }
                            })
                    else:
                        if self.config.data_type in ["orderbook", "all"]:
                            # Convert orderbooks to events
                            for _, row in orderbook_df.iterrows():
                                all_events.append({
                                    'timestamp': row['timestamp'],
                                    'type': 'orderbook',
                                    'exchange': exchange,
                                    'symbol': symbol,
                                    'data': row
                                })
                                
                        if self.config.data_type in ["tradebook", "all"]:
                            # Convert trades to events
                            for _, row in trades_df.iterrows():
                                all_events.append({
                                    'timestamp': row['timestamp'],
                                    'type': 'trade',
                                    'exchange': exchange,
                                    'symbol': symbol,
                                    'data': row
                                })
                        
                except Exception as e:
                    logger.error(f"Failed to load data for {exchange}:{symbol} - {e}")
                    
        # Sort by timestamp
        all_events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Prepared {len(all_events)} events for replay")
        return all_events
        
    async def _process_orderbook_event(self, event: Dict[str, Any]) -> None:
        """Process a single orderbook event"""
        data = event['data']
        
        # Convert to normalized format
        orderbook = self._create_normalized_orderbook(event)
        
        # Store in buffer
        key = f"{event['exchange']}:{event['symbol']}"
        self.orderbook_buffers[key].append(orderbook)
        
        # Call callbacks
        for callback in self.orderbook_callbacks:
            await callback(orderbook)
            
        # Extract features if enabled
        if self.config.enable_features:
            await self._extract_and_process_features(event['exchange'], event['symbol'])
            
        self.results.total_orderbooks += 1
        
    async def _process_trade_event(self, event: Dict[str, Any]) -> None:
        """Process a single trade event"""
        trade = self._create_normalized_trade(event)
        
        # Store in buffer
        key = f"{event['exchange']}:{event['symbol']}"
        self.trade_buffers[key].append(trade)
        
        # Call callbacks
        for callback in self.trade_callbacks:
            await callback(trade)
            
        self.results.total_trades += 1
        
    async def _extract_and_process_features(self, exchange: str, symbol: str) -> None:
        """Extract features and run anomaly detection"""
        logger.debug(f"Attempting to extract features for {symbol} on {exchange}") # Debug log
        start_time = datetime.utcnow()
        
        # Get current state
        key = f"{exchange}:{symbol}"
        orderbooks = list(self.orderbook_buffers[key])
        trades = list(self.trade_buffers[key])
        
        if not orderbooks and self.config.data_source != DataSource.LIVE_API: # Only require orderbooks if not live API
            logger.debug(f"Skipping feature extraction for {symbol} on {exchange}: No orderbooks available and not LIVE_API source.") # Debug log
            return {}, {}
        
        # Extract features
        features = {}
        
        # Cryptofeed features from orderbook (only if orderbook data is available)
        if orderbooks and 'cryptofeed' in self.feature_extractors:
            latest_ob = orderbooks[-1]
            ob_dict = {
                'bids': [[level.price, level.volume] for level in latest_ob.bids],
                'asks': [[level.price, level.volume] for level in latest_ob.asks]
            }
            cf_features = self.feature_extractors['cryptofeed'].extract(ob_dict)
            features.update({f'cf_{k}': v for k, v in cf_features.items()})
            logger.debug(f"Extracted cryptofeed features for {symbol} on {exchange}. Count: {len(cf_features)}") # Debug log
            
        # Tradebook features
        if trades and 'tradebook' in self.feature_extractors:
            trades_df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume,
                'side': t.side
            } for t in trades])
            
            tb_features = self.feature_extractors['tradebook'].extract(trades_df)
            features.update({f'tb_{k}': v for k, v in tb_features.items()})
            logger.debug(f"Extracted tradebook features for {symbol} on {exchange}. Count: {len(tb_features)}") # Debug log
            
        # Add metadata
        features['_timestamp'] = self.current_time
        features['_exchange'] = exchange
        features['_symbol'] = symbol
        
        # Track feature extraction time
        feature_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.results.feature_extraction_time_ms.append(feature_time)
        logger.debug(f"Feature extraction for {symbol} on {exchange} completed in {feature_time:.2f} ms.") # Debug log
        
        # Call feature callbacks
        for callback in self.feature_callbacks:
            await callback(features)
            
        # Initialize feature_stats to an empty dictionary
        feature_stats = {}

        # Calculate and store feature statistics
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if numeric_features:
            df = pd.DataFrame([numeric_features])
            if len(df) > 1:
                desc_stats = {
                    "mean": df.mean(axis=0).to_dict(),
                    "variance": df.var(axis=0).to_dict(),
                    "skewness": df.apply(lambda x: skew(x, nan_policy='omit')).to_dict(),
                    "kurtosis": df.apply(lambda x: kurtosis(x, nan_policy='omit')).to_dict()
                }
            else: # Handle single data point case
                desc_stats = {
                    "mean": df.mean(axis=0).to_dict(),
                    "variance": {k: 0.0 for k in numeric_features.keys()},
                    "skewness": {k: 0.0 for k in numeric_features.keys()},
                    "kurtosis": {k: 0.0 for k in numeric_features.keys()}
                }

            if exchange not in self.results.feature_statistics:
                self.results.feature_statistics[exchange] = {}
            self.results.feature_statistics[exchange][symbol] = desc_stats
            feature_stats = desc_stats # Assign desc_stats to feature_stats

        # Run anomaly detection if enabled
        if self.detection_system and self.config.enable_anomaly_detection:
            logger.debug(f"Calling anomaly detection for {symbol} on {exchange}.") # Debug log
            anomalies_result = await self._run_anomaly_detection(features)
        else:
            logger.debug(f"Anomaly detection skipped for {symbol} on {exchange}. Enabled: {self.config.enable_anomaly_detection}, System: {bool(self.detection_system)}") # Debug log
            anomalies_result = {} # Ensure anomalies_result is always a dictionary
        return features, feature_stats, anomalies_result # Ensure a return value at the end of the function
    async def _run_anomaly_detection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection on features"""
        logger.debug(f"Attempting to run anomaly detection for {features['_symbol']} on {features['_exchange']}") # Debug log
        start_time = datetime.utcnow()
        
        if not self.detection_system or not self.config.enable_anomaly_detection:
            return {} # Return empty dict if detection is not enabled or system not initialized
        anomalies = await self.detection_system.detect_anomalies(features)
        
        # Track detection time
        detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.results.anomaly_detection_time_ms.append(detection_time)
        logger.debug(f"Anomaly detection for {features['_symbol']} on {features['_exchange']} completed in {detection_time:.2f} ms. Severity: {anomalies['composite']['severity']}") # Debug log

        # Record anomalies
        if anomalies['composite']['severity'] != 'normal':
            self.results.total_anomalies += 1
            self.results.anomaly_timeline.append({
                'timestamp': self.current_time,
                'exchange': features['_exchange'],
                'symbol': features['_symbol'],
                'anomaly': anomalies
            })
            logger.info(f"Anomaly detected for {features['_symbol']} on {features['_exchange']}: {anomalies['composite']['severity']} - Detectors Flagged: {anomalies['composite']['num_detectors_flagged']}") # Info log for anomalies
            
        # Call anomaly callbacks
        for callback in self.anomaly_callbacks:
            await callback(anomalies)
        return anomalies # Ensure anomalies are returned
            
    def _create_normalized_orderbook(self, event: Dict[str, Any]) -> NormalizedOrderBook:
        """Create normalized orderbook from event data"""
        data = event['data']
        
        # Parse bid/ask data (format depends on data source)
        bids = []
        asks = []
        
        # Example format - adjust based on your data structure
        if 'bids' in data:
            bid_data = json.loads(data['bids']) if isinstance(data['bids'], str) else data['bids']
            bids = [OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in bid_data[:self.config.orderbook_depth]] # Ensure float conversion
            
        if 'asks' in data:
            ask_data = json.loads(data['asks']) if isinstance(data['asks'], str) else data['asks']
            asks = [OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in ask_data[:self.config.orderbook_depth]] # Ensure float conversion
            
        return NormalizedOrderBook(
            exchange=event['exchange'],
            symbol=event['symbol'],
            timestamp=event['timestamp'],
            bids=bids,
            asks=asks,
            sequence=data.get('nonce', 0)
        )
        
    def _create_normalized_trade(self, event: Dict[str, Any]) -> NormalizedTrade:
        """Create normalized trade from event data"""
        data = event['data']
        
        return NormalizedTrade(
            exchange=event['exchange'],
            symbol=event['symbol'],
            timestamp=data.get('timestamp', event['timestamp']), # Use event timestamp directly for OHLCV trades
            id=str(data.get('id', '')),
            price=float(data['price']), # Ensure float conversion
            volume=float(data.get('volume', data.get('amount'))), # Use 'amount' if 'volume' is not present, ensure float
            side=data.get('side', 'unknown'), # Default to 'unknown' if side is missing
            taker_side=data.get('taker_side', data.get('side', 'unknown')) # Use 'side' if 'taker_side' is missing
        )
        
    async def _save_results(self) -> None:
        """Save backtest results"""
        results_dir = Path(self.config.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{timestamp}.json"
        
        # Prepare results for serialization
        results_dict = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'symbols': self.config.symbols,
                'exchanges': self.config.exchanges,
                'replay_speed': self.config.replay_speed,
                'enable_features': self.config.enable_features,
                'enable_anomaly_detection': self.config.enable_anomaly_detection
            },
            'summary': {
                'start_time': self.results.start_time.isoformat(),
                'end_time': self.results.end_time.isoformat(),
                'duration_seconds': (self.results.end_time - self.results.start_time).total_seconds(),
                'total_orderbooks': self.results.total_orderbooks,
                'total_trades': self.results.total_trades,
                'total_anomalies': self.results.total_anomalies
            },
            'performance': {
                'avg_processing_time_ms': np.mean(self.results.processing_time_ms) if self.results.processing_time_ms else 0,
                'avg_feature_time_ms': np.mean(self.results.feature_extraction_time_ms) if self.results.feature_extraction_time_ms else 0,
                'avg_detection_time_ms': np.mean(self.results.anomaly_detection_time_ms) if self.results.anomaly_detection_time_ms else 0
            },
            'feature_statistics': self.results.feature_statistics, # NEW: Include feature statistics
            'anomaly_timeline': self.results.anomaly_timeline,
            'custom_metrics': self.results.custom_metrics
        }
        
        # Save to file
        with open(results_dir / filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
            
        logger.info(f"Saved backtest results to {results_dir / filename}")