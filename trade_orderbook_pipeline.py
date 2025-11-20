# trade_orderbook_pipeline.py
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import dataclasses # NEW: Import dataclasses module

# import uvloop
import ccxt # Import ccxt to catch specific exceptions
import yaml
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import make_asgi_app, Info
import grpc
from concurrent import futures
import pandas as pd


from core.exchange_manager_updated import ExchangeManager # Changed import
from core.data_loader import DataLoader
from core.exchange_interface import (
    NormalizedOrderBook, NormalizedTrade, NormalizedTicker
)


from features.extractors.advanced_cryptofeed_extractor import AdvancedCryptofeedExtractor
from features.extractors.advanced_tradebook_extractor import AdvancedTradebookExtractor
from features.extractors.enhanced_technical_extractor import EnhancedTechnicalExtractor
from features.extractors.dex_liquidity_extractor import DEXLiquidityExtractor
from features.extractors.futures_open_interest_funding_extractor import FuturesOpenInterestFundingExtractor
from features.extractors.cross_exchange_discrepancy_extractor import CrossExchangeDiscrepancyExtractor
from features.extractors.enhanced_options_extractor import EnhancedOptionsExtractor  # NEW IMPORT


from detection.system import DetectionSystem
from core.unified_output import write_unified_record # NEW IMPORTS


from core.logging_config import init_logging, get_logger # NEW import

# Global logger for initial setup messages, will be re-initialized within pipeline class
_global_logger = get_logger(__name__) # Renamed to avoid conflict with instance logger


# NEW IMPORTS FOR ALL EXCHANGES AND BACKTESTING
try:
    from adapters.cex_adapters import CEX_ADAPTERS, get_cex_adapter
    from adapters.dex_adapters import DEX_ADAPTERS, get_dex_adapter
    from backtesting.engine import BacktestConfig, BacktestingEngine, DataSource
    ALL_EXCHANGES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import exchange adapters or backtesting engine: {e}") # Use logging.error directly
    ALL_EXCHANGES_AVAILABLE = False
    CEX_ADAPTERS = {}
    DEX_ADAPTERS = {}


sys.path.insert(0, str(Path(__file__).parent / "adapters"))


info_metric = Info('trade_orderbook_pipeline', 'Trade and orderbook pipeline information')
info_metric.info({
    'version': '2.0.0',
    'start_time': datetime.now(timezone.utc).isoformat()
})


class TradeOrderbookPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # Initialize logging with settings from the loaded config
        init_logging(pipeline_config=self.config)
        self.logger = get_logger(__name__) # Re-get logger after config

        # NEW: Load exchanges config from JSON (assuming config_path is the JSON config)
        # The _load_config method already loads the entire config, so we can access it directly.
        exchanges_config_from_json = self.config.get("exchanges", {})

        self.exchange_manager = ExchangeManager(exchanges_config=exchanges_config_from_json, config_path=config_path) # Pass exchanges_config
        self.data_loader = DataLoader(self.exchange_manager, self.config.get('data_loader', {}))
        
        self.feature_extractors = self._initialize_feature_extractors()
        self.detection_system = DetectionSystem(pipeline_config=self.config) # Pass full config to DetectionSystem
        
        # NEW: Backtesting engine (initialized on demand)
        self.backtest_engine = None
        
        self.app = FastAPI(title="Trade & Orderbook Pipeline v2")
        self.websocket_clients: List[WebSocket] = []
        self._running = False
        self._setup_routes()
        self._setup_callbacks()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format. Please use .json, .yaml, or .yml.")
            
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        return {
            'cryptofeed': AdvancedCryptofeedExtractor(self.config.get('extractors', {}).get('cryptofeed', {})),
            'tradebook': AdvancedTradebookExtractor(self.config.get('extractors', {}).get('tradebook', {})),
            'technical': EnhancedTechnicalExtractor(self.config.get('extractors', {}).get('technical', {})),
            'dex': DEXLiquidityExtractor(self.config.get('extractors', {}).get('dex', {})),
            'futures': FuturesOpenInterestFundingExtractor(self.config.get('extractors', {}).get('futures', {})),
            'cross_exchange': CrossExchangeDiscrepancyExtractor(self.config.get('extractors', {}).get('cross_exchange', {}), self.exchange_manager),
            'options': EnhancedOptionsExtractor(self.config.get('extractors', {}).get('options', {}))  # NEW: OPTIONS EXTRACTOR
        }
        
    def _normalize_symbol_for_ccxt(self, symbol: str) -> str:
        """Converts a symbol to CCXT's BASE/QUOTE format if not already."""
        if '/' in symbol:
            return symbol.upper() # Ensure consistency
        # Attempt to infer BASE/QUOTE from common patterns (e.g., BTCUSDT -> BTC/USDT)
        # This is a simplified approach, a more robust solution might use exchange.load_markets()
        # and check against exchange.symbols
        if symbol.endswith('USDT'):
            return f"{symbol[:-4].upper()}/USDT"
        elif symbol.endswith('USD'):
            return f"{symbol[:-3].upper()}/USD"
        elif symbol.endswith('BTC'):
            return f"{symbol[:-3].upper()}/BTC"
        elif symbol.endswith('ETH'):
            return f"{symbol[:-3].upper()}/ETH"
        return symbol.upper() # Return as is, but uppercase

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            
        @self.app.get("/exchanges")
        async def get_exchanges():
            configured_exchanges = []
            for ex_id, ex_config in self.config.get('exchanges', {}).items():
                is_enabled = ex_config.get('enabled', False)
                is_connected = False
                if is_enabled and ex_id in self.exchange_manager.exchanges:
                    try:
                        is_connected = await self.exchange_manager.exchanges[ex_id].is_connected()
                    except Exception:
                        is_connected = False # Handle cases where is_connected might fail

                configured_exchanges.append({
                    "id": ex_id,
                    "enabled_in_config": is_enabled,
                    "is_active": is_connected # 'is_active' implies enabled AND connected
                })

            return JSONResponse(content={
                "configured_exchanges": configured_exchanges,
                "active_exchanges_count": len(self.exchange_manager.exchanges),
                "active_exchanges_list": list(self.exchange_manager.exchanges.keys())
            })
            
        @self.app.get("/instruments")
        async def get_instruments():
            return await self.exchange_manager.get_all_instruments()
            
        @self.app.get("/orderbook")
        async def get_orderbook_for_multiple_symbols(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get orderbooks for multiple symbols across all enabled exchanges."""
            all_orderbooks = {}
            for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
                exchange_orderbooks = {}
                for symbol_raw in symbols:
                    symbol = self._normalize_symbol_for_ccxt(symbol_raw)
                    try:
                        # Directly fetch orderbook data
                        ob = await exchange_adapter.fetch_order_book(symbol)
                        if ob:
                            normalized_ob = exchange_adapter._normalize_orderbook(ob, symbol)
                            exchange_orderbooks[symbol] = normalized_ob.dict()
                    except ccxt.NetworkError as e:
                        self.logger.warning(f"Network error fetching orderbook for {symbol} on {exchange_id}: {e}")
                    except ccxt.ExchangeError as e:
                        self.logger.warning(f"Exchange error fetching orderbook for {symbol} on {exchange_id}: {e}")
                    except Exception as e:
                        self.logger.warning(f"Unexpected error fetching orderbook for {symbol} on {exchange_id}: {e}")
                if exchange_orderbooks:
                    all_orderbooks[exchange_id] = exchange_orderbooks
            
            if not all_orderbooks:
                raise HTTPException(status_code=404, detail=f"No orderbooks found for any of the provided symbols on any enabled exchange.")
            return all_orderbooks
            
        @self.app.get("/trades")
        async def get_trades_for_multiple_symbols(symbols: List[str] = Query(..., description="Comma-separated list of symbols"), limit: int = 100):
            """Get recent trades for multiple symbols across all enabled exchanges."""
            all_trades = {}
            for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
                exchange_trades = {}
                for symbol_raw in symbols:
                    symbol = self._normalize_symbol_for_ccxt(symbol_raw)
                    try:
                        # Directly fetch trades data
                        trades = await exchange_adapter.fetch_trades(symbol, limit=limit)
                        if trades:
                            normalized_trades = [exchange_adapter._normalize_trade(t, symbol) for t in trades]
                            exchange_trades[symbol] = [t.dict() for t in normalized_trades]
                    except ccxt.NetworkError as e:
                        self.logger.warning(f"Network error fetching trades for {symbol} on {exchange_id}: {e}")
                    except ccxt.ExchangeError as e:
                        self.logger.warning(f"Exchange error fetching trades for {symbol} on {exchange_id}: {e}")
                    except Exception as e:
                        self.logger.warning(f"Unexpected error fetching trades for {symbol} on {exchange_id}: {e}")
                if exchange_trades:
                    all_trades[exchange_id] = exchange_trades
            
            if not all_trades:
                raise HTTPException(status_code=404, detail=f"No trades found for any of the provided symbols on any enabled exchange.")
            return all_trades
            

        @self.app.get("/aggregated")
        async def get_aggregated_orderbook(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get aggregated orderbook across exchanges for multiple symbols."""
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            
            all_aggregated_orderbooks = {}
            for symbol in normalized_symbols:
                # To aggregate, we need to ensure the data loader has current snapshots.
                # This implies a background polling mechanism or explicit fetch here.
                # Given the "fetch only when executed" request, we'll fetch for aggregation too.
                temp_orderbooks = {}
                for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
                    try:
                        ob = await exchange_adapter.fetch_order_book(symbol)
                        if ob:
                            normalized_ob = exchange_adapter._normalize_orderbook(ob, symbol)
                            # Store temporarily for aggregation
                            self.data_loader.orderbook_buffers[f"{exchange_id}:{symbol}"] = normalized_ob
                            temp_orderbooks[exchange_id] = normalized_ob
                    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                        self.logger.warning(f"Could not fetch orderbook for {symbol} on {exchange_id} for aggregation: {e}")

                if temp_orderbooks: # Only aggregate if some data was fetched
                    df = self.data_loader.get_aggregated_orderbook(symbol)
                    if not df.empty:
                        all_aggregated_orderbooks[symbol] = df.to_dict()
            
            if not all_aggregated_orderbooks:
                raise HTTPException(status_code=404, detail=f"No aggregated orderbooks found for the provided symbols.")
            return all_aggregated_orderbooks
            
        @self.app.get("/matrix")
        async def get_cross_exchange_matrices(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            """Get cross-exchange price matrices for multiple symbols."""
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]

            all_matrices = {}
            for symbol in normalized_symbols:
                # Similar to aggregation, fetch latest tickers for matrix calculation
                temp_tickers = {}
                for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
                    try:
                        ticker = await exchange_adapter.fetch_ticker(symbol)
                        if ticker:
                            normalized_ticker = exchange_adapter._normalize_ticker(ticker, symbol)
                            # Store temporarily for matrix calculation
                            self.data_loader.ticker_buffers[f"{exchange_id}:{symbol}"] = normalized_ticker # Store just the ticker
                            temp_tickers[exchange_id] = normalized_ticker
                    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                        self.logger.warning(f"Could not fetch ticker for {symbol} on {exchange_id} for matrix: {e}")

                if temp_tickers: # Only calculate if some data was fetched
                    df = self.data_loader.get_cross_exchange_matrix(symbol)
                    if not df.empty:
                        all_matrices[symbol] = df.to_dict()
            
            if not all_matrices:
                raise HTTPException(status_code=404, detail=f"No cross-exchange matrices found for the provided symbols.")
            return all_matrices
            
        @self.app.get("/features")
        async def get_features(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            self.logger.info(f"Received request for features for symbols: {normalized_symbols}")
            # Need to ensure data for features is fresh. Fetch trades and orderbooks.
            await self._fetch_data_for_features(normalized_symbols)
            features = await self._extract_all_features(normalized_symbols)
            self.logger.info(f"Returning features: {features}")
            return features
            
        @self.app.get("/anomalies")
        async def get_anomalies(symbols: List[str] = Query(..., description="Comma-separated list of symbols")):
            normalized_symbols = [self._normalize_symbol_for_ccxt(s) for s in symbols]
            self.logger.info(f"Received request for anomalies for symbols: {normalized_symbols}")
            # Need to ensure data for anomalies is fresh. Fetch trades and orderbooks.
            await self._fetch_data_for_features(normalized_symbols)
            all_features = await self._extract_all_features(normalized_symbols)
            self.logger.info(f"Features for anomalies: {all_features}")
            all_anomalies = {}
            for exchange_id, exchange_features in all_features.items():
                exchange_anomalies = {}
                for symbol, features in exchange_features.items():
                    try:
                        # Pass feature_names for meta_statistics
                        anomalies_result = await self.detection_system.detect_anomalies(features, feature_names=list(features.keys()))
                        
                        # Save unified record if in full mode
                        if self.config.get("output", {}).get("mode") == "full":
                            # Retrieve feature_stats from the _extract_features_for_single_exchange_symbol call
                            # all_features_output["feature_statistics"][exchange_id][symbol] was populated in _extract_all_features
                            feature_stats_for_unified_record = all_features_output.get("feature_statistics", {}).get(exchange_id, {}).get(symbol, {})
                            
                            write_unified_record(
                                config=self.config,
                                exchange=exchange_id,
                                symbol=symbol,
                                features=features,
                                feature_stats=feature_stats_for_unified_record,
                                detectors=anomalies_result["detectors"],
                                meta_stats=anomalies_result.get("meta_statistics", {}),
                                composite=anomalies_result["composite"]
                            )
                        
                        if anomalies_result:
                            exchange_anomalies[symbol] = anomalies_result
                    except Exception as e:
                        self.logger.warning(f"Could not detect anomalies for {symbol} on {exchange_id}: {e}")
                if exchange_anomalies:
                    all_anomalies[exchange_id] = exchange_anomalies
            
            if not all_anomalies:
                raise HTTPException(status_code=404, detail=f"No anomalies found for any of the provided symbols on any enabled exchange.")
            
            # Return full anomaly results if mode is "full", otherwise return a simplified summary
            if self.config.get("output", {}).get("mode") == "full":
                return JSONResponse(content=all_anomalies)
            else:
                # Return a simplified summary for "light" mode or if not configured
                simplified_anomalies = {}
                for ex_id, ex_anom in all_anomalies.items():
                    simplified_anomalies[ex_id] = {}
                    for sym, anom_res in ex_anom.items():
                        simplified_anomalies[ex_id][sym] = {
                            "composite_score": anom_res["composite"]["weighted_score"],
                            "severity": anom_res["composite"]["severity"],
                            "num_detectors_flagged": anom_res["composite"]["num_detectors_flagged"]
                        }
                return JSONResponse(content=simplified_anomalies)
            
        # NEW: Backtesting endpoints
        if ALL_EXCHANGES_AVAILABLE:
            @self.app.post("/backtest/run")
            async def run_backtest(
                symbols: List[str] = Query(..., description="Comma-separated list of symbols for backtesting"),
                start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
                end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
                start_time: Optional[str] = Query("00:00:00", description="Start time in HH:MM:SS format"),
                end_time: Optional[str] = Query("23:59:59", description="End time in HH:MM:SS format"),
                replay_speed: float = Query(10.0, description="Replay speed multiplier"),
                enable_features: bool = Query(True, description="Enable feature extraction during backtest"),
                enable_anomaly_detection: bool = Query(True, description="Enable anomaly detection during backtest"),
                data_type: str = Query("orderbook", description="Type of data to use: 'orderbook', 'tradebook', or 'both'")
            ):
                """Run a backtest with specified parameters, using all enabled exchanges and fetching historical data from Google Drive."""
                try:
                    # Automatically get enabled exchanges
                    enabled_exchanges = list(self.exchange_manager.exchanges.keys())
                    if not enabled_exchanges:
                        raise HTTPException(status_code=400, detail="No exchanges are currently enabled for backtesting.")
 
                    # Create backtest config
                    bt_config = BacktestConfig(
                        start_date=datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min).replace(tzinfo=timezone.utc),
                        end_date=datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.max).replace(tzinfo=timezone.utc),
                        start_time_str=start_time,
                        end_time_str=end_time,
                        symbols=symbols,
                        exchanges=enabled_exchanges, # Use all enabled exchanges
                        data_source=DataSource.GDRIVE, # Force GDRIVE data source
                        data_path="./historical_data", # Use default data path for Drive
                        gdrive_api_key=self.config.get('backtesting', {}).get('gdrive_api_key'),
                        gdrive_orderbook_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_root_id'),
                        gdrive_orderbook_binance_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_binance_root_id'),
                        gdrive_tradebook_root_id=self.config.get('backtesting', {}).get('gdrive_tradebook_root_id'),
                        data_type=data_type,
                        replay_speed=replay_speed,
                        enable_features=enable_features,
                        enable_anomaly_detection=enable_anomaly_detection,
                        output=self.config.get("output", {}) # Pass output configuration
                    )
                    
                    self.logger.info(f"Backtest config received: {dataclasses.asdict(bt_config)}") # Debug log
                    self.logger.info(f"enable_features: {enable_features}, enable_anomaly_detection: {enable_anomaly_detection}, data_type: {data_type}") # Debug log

                    # Create and run backtest
                    # Pass exchange_manager and data_loader to BacktestingEngine for live data fetching
                    self.backtest_engine = BacktestingEngine(bt_config, self.exchange_manager, self.data_loader)
                    results = await self.backtest_engine.run()
                    
                    return {
                        "status": "completed",
                        "summary": {
                            "total_orderbooks": results.total_orderbooks,
                            "total_trades": results.total_trades,
                            "total_anomalies": results.total_anomalies,
                            "duration": (results.end_time - results.start_time).total_seconds()
                        }
                    }
                except ValueError as ve:
                    self.logger.error(f"Date format error in backtest run: {ve}") # Added logging
                    raise HTTPException(status_code=400, detail=f"Date format error: {ve}. Please use YYYY-MM-DD.")
                except Exception as e:
                    self.logger.error(f"Error during backtest run: {e}", exc_info=True) # Added logging
                    raise HTTPException(status_code=400, detail=str(e))
                    
            @self.app.get("/backtest/scenarios")
            async def get_backtest_scenarios(
                symbols: Optional[List[str]] = Query(None, description="Comma-separated list of symbols for backtesting (optional)"),
                start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format (optional)"),
                end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format (optional)"),
                start_time: Optional[str] = Query("00:00:00", description="Start time in HH:MM:SS format (optional)"),
                end_time: Optional[str] = Query("23:59:59", description="End time in HH:MM:SS format (optional)"),
                replay_speed: float = Query(1.0, description="Replay speed multiplier"),
                enable_features: bool = Query(True, description="Enable feature extraction during backtest"),
                enable_anomaly_detection: bool = Query(True, description="Enable anomaly detection during backtest"),
                data_type: str = Query("orderbook", description="Type of data to use: 'orderbook', 'tradebook', or 'both'")
            ):
                """Get available backtest scenarios from config, or preview a custom one."""
                # If no parameters are provided, return the predefined scenarios
                if not symbols and not start_date and not end_date and not data_type and not start_time and not end_time:
                    self.logger.info("Returning predefined backtest scenarios.")
                    return self.config.get('backtesting', {}).get('scenarios', {})
 
                # Otherwise, construct and return a preview config
                try:
                    enabled_exchanges = list(self.exchange_manager.exchanges.keys())
                    if not enabled_exchanges:
                        self.logger.warning("No exchanges currently enabled for backtest scenario preview.") # Debug log
                        raise HTTPException(status_code=400, detail="No exchanges are currently enabled.")
 
                    # Use provided dates or defaults from config if available
                    parsed_start_date = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else datetime.fromisoformat(self.config.get('backtesting', {}).get('scenarios', {}).get('default', {}).get('start_date', '2025-01-01')).date()
                    parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.fromisoformat(self.config.get('backtesting', {}).get('scenarios', {}).get('default', {}).get('end_date', '2025-06-01')).date()
                    
                    bt_config = BacktestConfig(
                        start_date=datetime.combine(parsed_start_date, time.min).replace(tzinfo=timezone.utc),
                        end_date=datetime.combine(parsed_end_date, time.max).replace(tzinfo=timezone.utc),
                        start_time_str=start_time,
                        end_time_str=end_time,
                        symbols=symbols if symbols else self.config.get('symbols'), # Removed empty list default
                        exchanges=enabled_exchanges, # Use all enabled exchanges
                        gdrive_api_key=self.config.get('backtesting', {}).get('gdrive_api_key'),
                        gdrive_orderbook_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_root_id'),
                        gdrive_orderbook_binance_root_id=self.config.get('backtesting', {}).get('gdrive_orderbook_binance_root_id'),
                        gdrive_tradebook_root_id=self.config.get('backtesting', {}).get('gdrive_tradebook_root_id'),
                        data_type=data_type,
                        replay_speed=replay_speed,
                        enable_features=enable_features,
                        enable_anomaly_detection=enable_anomaly_detection,
                        data_source=DataSource.GDRIVE, # Force DRIVE data source
                        output=self.config.get("output", {}) # Pass output configuration
                    )
                    self.logger.info(f"Backtest scenario preview config: {dataclasses.asdict(bt_config)}") # Debug log
                    return bt_config.__dict__
                except ValueError as ve:
                    self.logger.error(f"Date format error in backtest scenario: {ve}") # Added logging
                    raise HTTPException(status_code=400, detail=f"Date format error: {ve}. Please use YYYY-MM-DD.")
                except Exception as e:
                    self.logger.error(f"Error during backtest scenario preview: {e}", exc_info=True) # Added logging
                    raise HTTPException(status_code=400, detail=str(e))
                
            @self.app.post("/exchanges/{exchange_id}/enable")
            async def enable_exchange(exchange_id: str):
                """Enable a specific exchange"""
                if exchange_id in CEX_ADAPTERS:
                    adapter = get_cex_adapter(exchange_id, self.config.get('exchanges', {}).get('cex', {}).get(exchange_id, {}))
                    self.exchange_manager.register_exchange(exchange_id, adapter)
                    await adapter.connect()
                    return {"status": "enabled", "exchange": exchange_id, "type": "cex"}
                elif exchange_id in DEX_ADAPTERS:
                    adapter = get_dex_adapter(exchange_id, self.config.get('exchanges', {}).get('dex', {}).get(exchange_id, {}))
                    self.exchange_manager.register_exchange(exchange_id, adapter)
                    await adapter.connect()
                    return {"status": "enabled", "exchange": exchange_id, "type": "dex"}
                else:
                    raise HTTPException(status_code=404, detail="Exchange not found")
                    
            @self.app.post("/exchanges/{exchange_id}/disable")
            async def disable_exchange(exchange_id: str):
                """Disable a specific exchange"""
                exchange = self.exchange_manager.exchanges.get(exchange_id)
                if exchange:
                    await exchange.disconnect()
                    del self.exchange_manager.exchanges[exchange_id]
                    return {"status": "disabled", "exchange": exchange_id}
                else:
                    raise HTTPException(status_code=404, detail="Exchange not enabled")
                    
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_clients.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except:
                self.websocket_clients.remove(websocket)
                
        metrics_app = make_asgi_app()
        self.app.mount("/metrics", metrics_app)
        
    def _setup_callbacks(self) -> None:
        async def broadcast_orderbook(orderbook: NormalizedOrderBook):
            if self.websocket_clients:
                message = {
                    "type": "orderbook",
                    "data": orderbook.dict()
                }
                disconnected_clients = []
                for client in self.websocket_clients:
                    try:
                        await client.send_json(message)
                    except:
                        disconnected_clients.append(client)
                        
                for client in disconnected_clients:
                    self.websocket_clients.remove(client)
                    
        async def broadcast_trade(trade: NormalizedTrade):
            if self.websocket_clients:
                message = {
                    "type": "trade",
                    "data": trade.dict()
                }
                disconnected_clients = []
                for client in self.websocket_clients:
                    try:
                        await client.send_json(message)
                    except:
                        disconnected_clients.append(client)
                        
                for client in disconnected_clients:
                    self.websocket_clients.remove(client)
                    
        self.exchange_manager.add_global_callback('orderbook', broadcast_orderbook)
        self.exchange_manager.add_global_callback('trade', broadcast_trade)
        
    async def _fetch_data_for_features(self, symbols: List[str], limit: int = 100) -> None:
        """Fetches latest data for feature extraction from all enabled exchanges."""
        fetch_tasks = []
        for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
            for symbol in symbols:
                # Fetch orderbook
                fetch_tasks.append(self.data_loader.fetch_and_store_orderbook(exchange_id, symbol))
                # Fetch trades
                fetch_tasks.append(self.data_loader.fetch_and_store_trades(exchange_id, symbol, limit))
                # Fetch ticker (if needed for features)
                fetch_tasks.append(self.data_loader.fetch_and_store_ticker(exchange_id, symbol))
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
    async def _extract_features_for_single_exchange_symbol(self, exchange: str, symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]: # Modified return type
        self.logger.info(f"Extracting features for {symbol} on {exchange}")
        all_features = {}
        
        # Ensure orderbook and trades data is available for feature extraction
        # These would have been fetched by _fetch_data_for_features if called from an API endpoint
        orderbook = self.data_loader.get_orderbook_snapshot(exchange, symbol)
        if orderbook:
            orderbook_dict = {
                'bids': [[float(level.price), float(level.volume)] for level in orderbook.bids],
                'asks': [[float(level.price), float(level.volume)] for level in orderbook.asks]
            }
            cryptofeed_features = self.feature_extractors['cryptofeed'].extract(orderbook_dict)
            all_features.update({f'cf_{k}': v for k, v in cryptofeed_features.items()})
            
        trades_data = self.data_loader.trade_buffers.get(f"{exchange}:{symbol}")
        if trades_data:
            try:
                trades_df = pd.DataFrame([t.dict() for t in trades_data])
                if not trades_df.empty:
                    tradebook_features = self.feature_extractors['tradebook'].extract(trades_df)
                    all_features.update({f'tb_{k}': v for k, v in tradebook_features.items()})
                else:
                    self.logger.warning(f"Trades DataFrame is empty for {symbol} on {exchange} after conversion.")
            except Exception as e:
                self.logger.error(f"Error creating trades DataFrame or extracting tradebook features for {symbol} on {exchange}: {e}")
            
        cross_exchange_data = await self._prepare_cross_exchange_data(symbol)
        self.logger.debug(f"Cross-exchange data for {symbol}: {cross_exchange_data}")
        if cross_exchange_data:
            cross_features = self.feature_extractors['cross_exchange'].extract(cross_exchange_data)
            self.logger.debug(f"Cross-exchange features for {symbol}: {cross_features}")
            all_features.update({f'cx_{k}': v for k, v in cross_features.items()})
            
        # NEW: Options features (if it's an option symbol)
        if self._is_option_symbol(symbol):
            options_data = await self._prepare_options_data(exchange, symbol, orderbook, trades_data) # Pass trades_data
            self.logger.debug(f"Options data for {symbol}: {options_data}")
            if options_data:
                options_features = self.feature_extractors['options'].extract(options_data)
                self.logger.debug(f"Options features for {symbol}: {options_features}")
                all_features.update({f'opt_{k}': v for k, v in options_features.items()})
                
        # NEW: DEX features (if applicable and modules available)
        if ALL_EXCHANGES_AVAILABLE and exchange in DEX_ADAPTERS:
            dex_exchange = self.exchange_manager.exchanges.get(exchange)
            if dex_exchange and hasattr(dex_exchange, 'get_pool_data'):
                try:
                    pool_data = await dex_exchange.get_pool_data(symbol)
                    self.logger.debug(f"DEX pool data for {symbol} on {exchange}: {pool_data}")
                    dex_features = self.feature_extractors['dex'].extract(pool_data)
                    self.logger.debug(f"DEX features for {symbol} on {exchange}: {dex_features}")
                    all_features.update({f'dex_{k}': v for k, v in dex_features.items()})
                except Exception as e:
                    self.logger.warning(f"Could not get DEX pool data for {symbol} on {exchange}: {e}")
        # Calculate feature statistics
        feature_stats = {}
        numeric_features = {k: v for k, v in all_features.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if numeric_features:
            df = pd.DataFrame([numeric_features])
            feature_stats = {
                "mean": df.mean(axis=0).to_dict(),
                "variance": {},
                "skewness": {},
                "kurtosis": {}
            }
            if len(df) <= 1:
                for col in df.columns:
                    feature_stats["variance"][col] = 0.0
                    feature_stats["skewness"][col] = 0.0
                    feature_stats["kurtosis"][col] = 0.0
            else:
                feature_stats["variance"] = df.var(axis=0).to_dict()
                feature_stats["skewness"] = df.apply(lambda x: skew(x, nan_policy='omit')).to_dict()
                feature_stats["kurtosis"] = df.apply(lambda x: kurtosis(x, nan_policy='omit')).to_dict()
        # Calculate feature statistics
        
        # Calculate feature statistics
        
        # Save raw features and statistics if enabled
        # The save_feature_outputs function is now removed from here.
        # Feature statistics will be returned directly and handled by the BacktestingEngine.
        # Calculate feature statistics
        
        

        self.logger.info(f"Finished extracting features for {symbol} on {exchange}. Total features: {len(all_features)}")
        return all_features, feature_stats # Return both features and feature_stats
        
    async def _extract_all_features(self, symbols: List[str]) -> Dict[str, Any]:
        """Extract all features for multiple symbols across all enabled exchanges."""
        all_features_output = {"features": {}, "feature_statistics": {}} # Initialize with nested dictionaries
        # Ensure data is fetched before extracting features
        # This will be handled by the calling API endpoint's _fetch_data_for_features
        for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
            all_features_output["features"][exchange_id] = {}
            all_features_output["feature_statistics"][exchange_id] = {}
            for symbol in symbols:
                try:
                    features, feature_stats = await self._extract_features_for_single_exchange_symbol(exchange_id, symbol)
                    if features:
                        all_features_output["features"][exchange_id][symbol] = features
                        all_features_output["feature_statistics"][exchange_id][symbol] = feature_stats
                except Exception as e:
                    self.logger.warning(f"Could not extract features for {symbol} on {exchange_id}: {e}")
        return all_features_output
        
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an option (contains strike and C/P)"""
        # Common patterns: BTC-28MAR25-50000-C, ETH-3000-C-280325
        parts = symbol.split('-')
        if len(parts) >= 4:
            # Check if last part is C or P
            if parts[-1].upper() in ['C', 'P', 'CALL', 'PUT']:
                return True
        # Also check for other patterns
        if any(x in symbol.upper() for x in ['-C-', '-P-', 'CALL', 'PUT']):
            return True
        return False
        
    async def _prepare_options_data(self, exchange: str, symbol: str, 
                                  orderbook: Optional[NormalizedOrderBook],
                                  trades_data: List[NormalizedTrade]) -> Dict[str, Any]: # Changed trades_df to trades_data
        """Prepare options data for feature extraction"""
        options_data = {
            'symbol': symbol,
            'exchange': exchange,
            'orderbook': {
                'bids': [[level.price, level.volume] for level in orderbook.bids] if orderbook else [],
                'asks': [[level.price, level.volume] for level in orderbook.asks] if orderbook else []
            } if orderbook else {}
        }
        
        # Parse option details from symbol
        parts = symbol.split('-')
        if len(parts) >= 4:
            # Try to extract strike price
            for part in parts:
                try:
                    strike = float(part)
                    options_data['strike'] = strike
                    break
                except ValueError:
                    continue
                    
            # Determine option type
            if parts[-1].upper() in ['C', 'CALL']:
                options_data['option_type'] = 'call'
            elif parts[-1].upper() in ['P', 'PUT']:
                options_data['option_type'] = 'put'
                
        # Get underlying price (try to fetch spot price)
        underlying_symbol = self._get_underlying_symbol(symbol)
        underlying_ob = self.data_loader.get_orderbook_snapshot(exchange, underlying_symbol)
        if underlying_ob and underlying_ob.bids and underlying_ob.asks:
            options_data['underlying_price'] = (underlying_ob.bids[0].price + underlying_ob.asks[0].price) / 2
            
        # Add market data
        if orderbook and orderbook.bids and orderbook.asks:
            options_data['bid'] = orderbook.bids[0].price
            options_data['ask'] = orderbook.asks[0].price
            options_data['mid_price'] = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
            
        # Add recent trades
        if trades_data: # Check if trades_data is not empty
            trades_df = pd.DataFrame([t.dict() for t in trades_data]) # Convert to DataFrame here
            if not trades_df.empty:
                options_data['recent_trades'] = trades_df.to_dict('records')
                options_data['last_price'] = trades_df.iloc[-1]['price']
                options_data['volume'] = trades_df['volume'].sum()
            
        # Get ticker data if available
        ticker_key = f"{exchange}:{symbol}"
        if ticker_key in self.data_loader.ticker_buffers:
            latest_ticker = self.data_loader.ticker_buffers[ticker_key] # Directly get NormalizedTicker
            options_data['volume_24h'] = latest_ticker.volume_24h
            options_data['open_interest'] = getattr(latest_ticker, 'open_interest', 0)
                
        # Add expiry parsing
        options_data['expiry'] = self._parse_expiry_from_symbol(symbol)
        
        return options_data
        
    def _get_underlying_symbol(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        # BTC-28MAR25-50000-C -> BTC/USDT
        parts = option_symbol.split('-')
        if parts:
            base = parts[0]
            # Common mappings
            if base in ['BTC', 'ETH', 'SOL', 'MATIC']:
                return f"{base}/USDT"
        return "BTC/USDT"  # Default
        
    def _parse_expiry_from_symbol(self, symbol: str) -> str:
        """Parse expiry date from option symbol"""
        # Look for date patterns like 28MAR25, 280325, etc.
        import re
        
        # Pattern 1: DDMMMYY (28MAR25)
        pattern1 = r'(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})'
        match1 = re.search(pattern1, symbol.upper())
        if match1:
            day, month, year = match1.groups()
            month_num = {
                'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
            }[month]
            return f"20{year}-{month_num}-{day.zfill(2)}"
            
        # Pattern 2: YYMMDD (250328)
        pattern2 = r'(\d{2})(\d{2})(\d{2})'
        match2 = re.search(pattern2, symbol)
        if match2:
            year, month, day = match2.groups()
            return f"20{year}-{month}-{day}"
            
        # Default to 30 days from now
        return (datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d')
        
    async def _prepare_cross_exchange_data(self, symbol: str) -> Dict[str, Any]:
        data = {}
        
        # Fetch latest ticker data for all exchanges for cross-exchange analysis
        fetch_tasks = []
        for exchange_id, exchange_adapter in self.exchange_manager.exchanges.items():
            fetch_tasks.append(self.data_loader.fetch_and_store_ticker(exchange_id, symbol))
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        for exchange_id in self.exchange_manager.exchanges.keys():
            ticker = self.data_loader.ticker_buffers.get(f"{exchange_id}:{symbol}")
            if ticker: # ticker is now directly NormalizedTicker
                data[exchange_id] = {
                    'price': ticker.last,
                    'volume_24h': ticker.volume_24h,
                    'timestamp': ticker.timestamp,
                    'orderbook': self.data_loader.get_orderbook_snapshot(exchange_id, symbol) # This would be populated by _fetch_and_store_orderbook
                }
                
        return data
        
    async def start(self) -> None:
        self._running = True
        
        # Auto-register exchanges
        await self._auto_register_exchanges()
        
        await self.data_loader.initialize()
        
        await self.exchange_manager.start_all() # Connects to exchanges
        
        # No automatic symbol subscription/polling on startup
        
        self.logger.info(f"Trade & Orderbook Pipeline v2 started with {len(self.exchange_manager.exchanges)} exchanges")
        if ALL_EXCHANGES_AVAILABLE:
            self.logger.info("All exchange adapters loaded - CEX and DEX support available")
            self.logger.info("Backtesting support available")
            self.logger.info("Options analytics support enabled")
        else:
            self.logger.info("Running with original exchange adapters only")
            self.logger.info("To enable all exchanges: pip install ccxt web3 pyarrow")
            self.logger.info("To enable options: pip install mibian py_vollib QuantLib")
        
    async def _auto_register_exchanges(self) -> None:
        """Auto-register all available exchanges from config"""
        all_exchanges_config = self.config.get('exchanges', {})
        
        for exchange_id, exchange_config in all_exchanges_config.items():
            if exchange_config.get('enabled', False):
                try:
                    if exchange_id in CEX_ADAPTERS:
                        adapter = get_cex_adapter(exchange_id, exchange_config)
                        self.exchange_manager.register_exchange(exchange_id, adapter)
                        self.logger.info(f"Registered CEX: {exchange_id}")
                    elif exchange_id in DEX_ADAPTERS:
                        adapter = get_dex_adapter(exchange_id, exchange_config)
                        self.exchange_manager.register_exchange(exchange_id, adapter)
                        self.logger.info(f"Registered DEX: {exchange_id}")
                    else:
                        self.logger.warning(f"No known adapter for enabled exchange: {exchange_id}. Skipping.")
                except Exception as e:
                    self.logger.error(f"Failed to register exchange {exchange_id}: {e}")
                    
    # Removed _fetch_and_subscribe_dynamic_symbols as per user request
    # Data will now be fetched on demand via API endpoints.
 
    async def stop(self) -> None:
        self._running = False
 
        await self.exchange_manager.stop_all()
        await self.data_loader.close()
 
        self.logger.info("Trade & Orderbook Pipeline v2 stopped")
 
 
import platform
 
async def main():
    # Use uvloop only if not on Windows
    if platform.system() != "Windows":
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    else:
        _global_logger.info("Running on Windows: using default asyncio event loop")
 
    pipeline = TradeOrderbookPipeline("config/pipeline_config.json")
 
    loop = asyncio.get_event_loop() # Ensure this is the correct way to get the event loop
 
    def signal_handler(sig, frame):
        _global_logger.info("Shutting down...")
        asyncio.create_task(pipeline.stop())
        loop.stop()
 
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
 
    await pipeline.start()
 
    server_config = uvicorn.Config(
        app=pipeline.app,
        host="0.0.0.0",
        port=pipeline.config.get('server', {}).get('port', 8000),
        log_level=pipeline.config.get('logging', {}).get('level', 'info').lower()
    )
    server = uvicorn.Server(server_config)
 
    try:
        await server.serve()
    except OSError as e:
        _global_logger.error(f"Failed to start Uvicorn server: {e}. This usually means the port is already in use. Please ensure no other instances of the pipeline are running.")
    except Exception as e:
        _global_logger.error(f"An unexpected error occurred while running the Uvicorn server: {e}")
 
 
if __name__ == "__main__":
    asyncio.run(main())

