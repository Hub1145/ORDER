# core/exchange_manager.py
"""Exchange manager for handling multiple exchanges"""


import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import yaml
import importlib
import sys
import json

from core.exchange_interface import ExchangeInterface


logger = logging.getLogger(__name__)


class ExchangeManager:
    """Manages multiple exchange connections"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.exchanges: Dict[str, ExchangeInterface] = {}
        self.config = self._load_config(config_path) if config_path else {}
        self.callbacks = {
            'orderbook': [],
            'trade': [],
            'ticker': []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Only JSON config is supported. Please use .json config file.")
            
    def register_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Register an exchange adapter"""
        if exchange_id in self.exchanges:
            logger.warning(f"Exchange {exchange_id} already registered, replacing")
        self.exchanges[exchange_id] = exchange
        logger.info(f"Registered exchange: {exchange_id}")
        
    def auto_register_exchanges(self) -> None:
        """Auto-register exchanges from config"""
        # Load adapters from the adapters directory
        adapters_dir = Path(__file__).parent.parent / "adapters"
        if not adapters_dir.exists():
            logger.warning("Adapters directory not found")
            return
            
        sys.path.insert(0, str(adapters_dir))
        
        # Register configured exchanges
        exchanges_config = self.config.get('exchanges', {})
        
        for exchange_id, exchange_config in exchanges_config.items():
            if not exchange_config.get('enabled', True):
                continue
            
            # Skip 'cex' and 'dex' as they are handled by trade_orderbook_pipeline.py's _auto_register_exchanges
            if exchange_id in ['cex', 'dex']:
                logger.debug(f"Skipping auto-registration for {exchange_id} as it's handled by specific CEX/DEX logic.")
                continue

            try:
                # Try to import the adapter
                module_name = f"{exchange_id}_adapter"
                module = importlib.import_module(module_name)
                
                # Look for adapter class
                adapter_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, ExchangeInterface) and
                        attr != ExchangeInterface):
                        adapter_class = attr
                        break
                        
                if adapter_class:
                    adapter = adapter_class(exchange_id, exchange_config)
                    self.register_exchange(exchange_id, adapter)
                else:
                    logger.warning(f"No adapter class found in {module_name}")
                    
            except ImportError as e:
                logger.error(f"Failed to import adapter for {exchange_id}: {e}. Ensure the adapter file exists and is correctly named.")
            except Exception as e:
                logger.error(f"Error auto-registering exchange {exchange_id}: {e}")
                
    async def start_all(self) -> None:
        """Start all registered exchanges"""
        tasks = []
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._start_exchange(exchange_id, exchange))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _start_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Start a single exchange"""
        try:
            await exchange.connect()
            
            # Register global callbacks
            for event_type, callbacks in self.callbacks.items():
                for callback in callbacks:
                    exchange.add_callback(event_type, callback)
                    
            logger.info(f"Started exchange: {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to start exchange {exchange_id}: {e}")
            
    async def stop_all(self) -> None:
        """Stop all exchanges"""
        tasks = []
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._stop_exchange(exchange_id, exchange))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _stop_exchange(self, exchange_id: str, exchange: ExchangeInterface) -> None:
        """Stop a single exchange"""
        try:
            await exchange.disconnect()
            logger.info(f"Stopped exchange: {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to stop exchange {exchange_id}: {e}")
            
    def add_global_callback(self, event_type: str, callback: Callable) -> None:
        """Add a callback for all exchanges"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        
        # Add to already connected exchanges
        for exchange in self.exchanges.values():
            exchange.add_callback(event_type, callback)
            
    def get_exchange(self, exchange_id: str) -> Optional[ExchangeInterface]:
        """Get a specific exchange"""
        return self.exchanges.get(exchange_id)
        
    async def get_all_instruments(self) -> Dict[str, List[str]]:
        """Get instruments from all exchanges"""
        instruments = {}
        for exchange_id, exchange in self.exchanges.items():
            try:
                exchange_instruments = await exchange.get_instruments()
                instruments[exchange_id] = exchange_instruments
            except Exception as e:
                logger.error(f"Failed to get instruments from {exchange_id}: {e}")
                instruments[exchange_id] = []
        return instruments