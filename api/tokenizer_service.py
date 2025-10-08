import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenizerService:
    
    def __init__(self, tokenizer_dir: str = "code/data/tokenized_data"):
        self.tokenizers = {}
        self.tokenizer_dir = Path(tokenizer_dir)
        self.default_window = "30s"
        self.time_windows = ["10s", "30s", "1min"]
        self._load_tokenizers()
    
    def _load_tokenizers(self):
        """Load tokenizers for all time windows from environment variables or default paths """
        
        # Adding code path for imports
        import sys
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
        
        from src.data.tokenization import NetworkTokenizer, TokenType, TokenMetadata
        
        tokenizer_paths = {
            "10s": os.environ.get("TOKENIZER_10S_PATH", "code/data/tokenized_data/tokenizer_10s.pkl"),
            "30s": os.environ.get("TOKENIZER_30S_PATH", "code/data/tokenized_data/tokenizer_30s.pkl"),
            "1min": os.environ.get("TOKENIZER_1min_PATH", "code/data/tokenized_data/tokenizer_1min.pkl")
        }
        
        for window, tokenizer_path in tokenizer_paths.items():
            tokenizer_path = Path(tokenizer_path)
            
            if tokenizer_path.exists():
                try:
                    tokenizer = NetworkTokenizer(config={'vocab_size': 10000})
                    tokenizer.load_tokenizer(str(tokenizer_path))
                    self.tokenizers[window] = tokenizer
                    logger.info(f"[{window}] Tokenizer loaded successfully from {tokenizer_path}")
                except Exception as e:
                    logger.error(f"[{window}] Failed to load tokenizer: {e}")
            else:
                logger.warning(f"[{window}] Tokenizer not found at {tokenizer_path}")
        
        # Create fallback tokenizer if none loaded
        if not self.tokenizers:
            logger.warning("No tokenizers loaded, creating fallback tokenizer")
            self.tokenizers['fallback'] = self._create_fallback_tokenizer()
        
        logger.info(f"Loaded tokenizers for time windows: {list(self.tokenizers.keys())}")
    
    def _create_fallback_tokenizer(self):
        """Create a fallback tokenizer that generates random tokens """
        class FallbackTokenizer:
            def __init__(self):
                self.vocab_size = 10000
                self.max_length = 256
            
            def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
                n_samples = len(df)
                return {
                    'input_ids': np.random.randint(0, self.vocab_size, (n_samples, self.max_length)),
                    'attention_mask': np.ones((n_samples, self.max_length)),
                    'token_type_ids': np.zeros((n_samples, self.max_length))
                }
        
        return FallbackTokenizer()
    
    def is_ready(self) -> bool:
        """Check if any tokenizers are loaded """
        return len(self.tokenizers) > 0
    
    def get_available_time_windows(self) -> list[str]:
        """Get list of time windows with loaded tokenizers"""
        return [window for window in self.tokenizers.keys() if window != 'fallback']
    
    def get_tokenizer_info_for_window(self, time_window: str) -> Dict[str, Any]:
        """Get information about a specific time window tokenizer"""
        return {
            "time_window": time_window,
            "loaded": time_window in self.tokenizers,
            "vocab_size": getattr(self.tokenizers.get(time_window), 'vocab_size', 0),
            "max_length": getattr(self.tokenizers.get(time_window), 'max_length', 0),
            "is_fallback": time_window not in self.tokenizers or time_window == 'fallback'
        }
    
    def tokenize(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        time_window: str = "30s",
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """Tokenize data using time window specific tokenizer """
        
        selected_tokenizer = None
        selected_window = None
        
        if time_window in self.tokenizers:
            selected_tokenizer = self.tokenizers[time_window]
            selected_window = time_window
            logger.debug(f"Using tokenizer for requested time window: {time_window}")
        elif self.default_window in self.tokenizers:
            selected_tokenizer = self.tokenizers[self.default_window]
            selected_window = self.default_window
            logger.warning(f"Using default tokenizer ({self.default_window}) for requested {time_window}")
        else:
            available_windows = [w for w in self.tokenizers.keys() if w != 'fallback']
            if available_windows:
                selected_window = available_windows[0]
                selected_tokenizer = self.tokenizers[selected_window]
                logger.warning(f"Using available tokenizer ({selected_window}) for requested {time_window}")
            else:
                selected_tokenizer = self.tokenizers.get('fallback', self._create_fallback_tokenizer())
                selected_window = 'fallback'
                logger.warning(f"Using fallback tokenizer for {time_window}")
        
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
            else:
                columns = ['feature_0']
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data, columns=columns)
            
            if 'datetime' not in df.columns:
                df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
        else:
            df = data.copy()
        
        required_columns = ['packet_count_sum', 'total_bytes_sum', 'packet_rate_mean', 
                           'byte_rate_mean', 'mean_packet_size_mean']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.random.uniform(0, 1000, len(df))
        
        # Tokenize
        try:
            result = selected_tokenizer.transform(df)
            
            if isinstance(result, dict):
                result['tokenizer_info'] = {
                    'requested_window': time_window,
                    'used_window': selected_window,
                    'vocab_size': getattr(selected_tokenizer, 'vocab_size', 10000),
                    'max_length': getattr(selected_tokenizer, 'max_length', 256),
                    'is_fallback': selected_window == 'fallback'
                }
            
            if not return_attention_mask and 'attention_mask' in result:
                del result['attention_mask']
            
            logger.debug(f"Tokenized {len(df)} samples using {selected_window} tokenizer")
            return result
            
        except Exception as e:
            logger.error(f"Tokenization failed with {selected_window} tokenizer: {e}")
            
            # Fallback to simple random tokenization
            n_samples = len(df)
            fallback_result = {
                'input_ids': np.random.randint(0, 10000, (n_samples, 256)),
                'tokenizer_info': {
                    'requested_window': time_window,
                    'used_window': 'emergency_fallback',
                    'vocab_size': 10000,
                    'max_length': 256,
                    'is_fallback': True,
                    'error': str(e)
                }
            }
            
            if return_attention_mask:
                fallback_result['attention_mask'] = np.ones((n_samples, 256))
            
            return fallback_result
    
    def get_vocab_size(self, time_window: str = None) -> int:
        """Get vocabulary size for a specific time window """
        if time_window and time_window in self.tokenizers:
            tokenizer = self.tokenizers[time_window]
        elif self.tokenizers:
            tokenizer = list(self.tokenizers.values())[0]
        else:
            return 10000
        
        return getattr(tokenizer, 'vocab_size', 10000)
    
    def get_max_length(self, time_window: str = None) -> int:
        """Get max sequence length for a specific time window"""
        if time_window and time_window in self.tokenizers:
            tokenizer = self.tokenizers[time_window]
        elif self.tokenizers:
            tokenizer = list(self.tokenizers.values())[0]
        else:
            return 256
        
        return getattr(tokenizer, 'max_length', 256)
    
    def get_tokenizer_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all tokenizers """
        status = {}
        
        for window in self.time_windows:
            status[window] = self.get_tokenizer_info_for_window(window)
        
        if 'fallback' in self.tokenizers:
            status['fallback'] = self.get_tokenizer_info_for_window('fallback')
        
        return status
    
    def reload_tokenizer(self, time_window: str, tokenizer_path: str = None) -> bool:
        """Reload a specific time window tokenizer"""
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
            
            from src.data.tokenization import NetworkTokenizer
            
            if tokenizer_path is None:
                # Use environment variable or default path
                env_key = f"TOKENIZER_{time_window.upper().replace('MIN', 'min')}_PATH"
                tokenizer_path = os.environ.get(env_key, f"code/data/tokenized_data/tokenizer_{time_window}.pkl")
            
            tokenizer_path = Path(tokenizer_path)
            
            if not tokenizer_path.exists():
                logger.error(f"Tokenizer file not found: {tokenizer_path}")
                return False
            
            tokenizer = NetworkTokenizer(config={'vocab_size': 10000})
            tokenizer.load_tokenizer(str(tokenizer_path))
            
            self.tokenizers[time_window] = tokenizer
            logger.info(f"[{time_window}] Tokenizer reloaded successfully from {tokenizer_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{time_window}] Failed to reload tokenizer: {e}")
            return False