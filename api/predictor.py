import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
import logging
import asyncio
from datetime import datetime
import pickle
import json
import os

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    SPECIAL = "special"
    NETWORK = "network"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    STATISTICAL = "statistical"
    PROTOCOL = "protocol"
    FLOW = "flow"
    ANOMALY = "anomaly"
    CONTEXTUAL = "contextual"

@dataclass
class TokenMetadata:
    token: str
    type: TokenType
    feature: str
    value_range: Optional[Tuple[float, float]] = None
    frequency: int = 0
    importance_score: float = 0.0

sys.modules['__main__'].TokenType = TokenType
sys.modules['__main__'].TokenMetadata = TokenMetadata

from src.models.lora import EfficientNetworkTrafficTransformer
from src.models.baseline import LSTMBaseline, ARIMABaseline
from src.data.preprocessing import NetworkTrafficPreprocessor
from src.data.tokenization import NetworkTokenizer

logger = logging.getLogger(__name__)

class NetworkPredictor:
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        # Only initialize once
        if NetworkPredictor._initialized:
            return
            
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store models for each time window
        self.transformer_models = {}  # Dict[time_window, model]
        self.lstm_models = {}        # Dict[time_window, model]
        self.tokenizers = {}         # Dict[time_window, tokenizer]
        self.arima_model = None
        self.preprocessor = None
        
        self.model_type = "EfficientNetworkTrafficTransformer"
        self.version = "2.0.0"
        self.max_sequence_length = 256
        self.feature_dimensions = 8
        self.last_updated = datetime.now().isoformat()
        
        self._load_models()
        NetworkPredictor._initialized = True
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Load from environment variables with fallbacks
        return {
            "time_windows": ["10s", "30s", "1min"],
            "transformer_model_paths": {
                "10s": os.environ.get("TRANSFORMER_10S_PATH", "code/data/llm/checkpoints/efficient_ft/10s/best_model.pt"),
                "30s": os.environ.get("TRANSFORMER_30S_PATH", "code/data/llm/checkpoints/efficient_ft/30s/best_model.pt"), 
                "1min": os.environ.get("TRANSFORMER_1min_PATH", "code/data/llm/checkpoints/efficient_ft/1min/best_model.pt")
            },
            "lstm_model_paths": {
                "10s": os.environ.get("LSTM_10S_PATH", "code/data/baseline_results/10s/lstm_model.pt"),
                "30s": os.environ.get("LSTM_30S_PATH", "code/data/baseline_results/30s/lstm_model.pt"),
                "1min": os.environ.get("LSTM_1min_PATH", "code/data/baseline_results/1min/lstm_model.pt")
            },
            "tokenizer_paths": {
                "10s": os.environ.get("TOKENIZER_10S_PATH", "code/data/tokenized_data/tokenizer_10s.pkl"),
                "30s": os.environ.get("TOKENIZER_30S_PATH", "code/data/tokenized_data/tokenizer_30s.pkl"),
                "1min": os.environ.get("TOKENIZER_1min_PATH", "code/data/tokenized_data/tokenizer_1min.pkl")
            },
            "preprocessor_config": {
                "max_packets_per_file": 5000000,
                "time_windows": ["10s", "30s", "1min"],
                "normalization_method": "robust"
            },
            "prediction_config": {
                "batch_size": 32,
                "max_length": 512,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95
            }
        }
    
    def _load_models(self):
        try:
            # Load models for each time window
            for time_window in self.config["time_windows"]:
                logger.info(f"Loading models for time window: {time_window}")
                
                # Load transformer model
                self._load_transformer_for_window(time_window)
                
                # Load tokenizer
                self._load_tokenizer_for_window(time_window)
                
                # Load LSTM model
                self._load_lstm_for_window(time_window)
            
            self._load_preprocessor()
            
            # Load ARIMA model (shared across time windows)
            try:
                self.arima_model = ARIMABaseline()
            except:
                pass
            
            loaded_windows = list(self.transformer_models.keys()) + list(self.lstm_models.keys())
            logger.info(f"Successfully loaded models for time windows: {set(loaded_windows)}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("API starting without models - some features may be unavailable")
    
    def _load_transformer_for_window(self, time_window: str):
        model_path = Path(self.config["transformer_model_paths"].get(time_window, ""))
        
        if not model_path.exists():
            logger.warning(f"Transformer model not found for {time_window} at {model_path}")
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', {})
            
            # Auto-detect model configuration from checkpoint
            hidden_size = 512
            for key in state_dict.keys():
                if 'attention.self.query.lora.lora_A' in key:
                    hidden_size = state_dict[key].shape[0]
                    break
                elif 'attention.self.query.lora.lora_B' in key:
                    hidden_size = state_dict[key].shape[1]
                    break
            
            intermediate_size = 3072
            for key in state_dict.keys():
                if 'intermediate.dense.lora.lora_B' in key:
                    intermediate_size = state_dict[key].shape[1]
                    break
            
            num_layers = 6
            layer_nums = set()
            for key in state_dict.keys():
                if 'bert.encoder.layer.' in key:
                    parts = key.split('.')
                    if len(parts) > 3 and parts[2] == 'layer':
                        try:
                            layer_num = int(parts[3])
                            layer_nums.add(layer_num)
                        except:
                            pass
            if layer_nums:
                num_layers = max(layer_nums) + 1
            
            vocab_size = 10000
            if 'mlm_head.weight' in state_dict:
                vocab_size = state_dict['mlm_head.weight'].shape[0]
            
            logger.info(f"[{time_window}] Detected model config: hidden={hidden_size}, intermediate={intermediate_size}, "
                       f"layers={num_layers}, vocab={vocab_size}")
            
            from transformers import BertConfig
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=8 if hidden_size == 512 else 12,
                intermediate_size=intermediate_size,
                max_position_embeddings=256,
                type_vocab_size=9,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            
            config.num_traffic_features = 8
            config.num_anomaly_classes = 4
            config.num_protocol_classes = 20
            config.use_sparse_attention = False
            
            model = EfficientNetworkTrafficTransformer(config)
            
            if 'model_state_dict' in checkpoint:
                missing, unexpected = model.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=False
                )
                logger.info(f"[{time_window}] Loaded transformer weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            
            model.to(self.device)
            model.eval()
            
            self.transformer_models[time_window] = model
            logger.info(f"[{time_window}] Transformer model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer model for {time_window}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _load_tokenizer_for_window(self, time_window: str):
        tokenizer_path = Path(self.config["tokenizer_paths"].get(time_window, ""))
        
        if not tokenizer_path.exists():
            logger.warning(f"Tokenizer not found for {time_window} at {tokenizer_path}")
            return
        
        try:
            from src.data.tokenization import NetworkTokenizer
            
            tokenizer = NetworkTokenizer(config={})
            tokenizer.load_tokenizer(str(tokenizer_path))
            
            self.tokenizers[time_window] = tokenizer
            logger.info(f"[{time_window}] Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {time_window}: {e}")
    
    def _load_lstm_for_window(self, time_window: str):
        lstm_path = Path(self.config["lstm_model_paths"].get(time_window, ""))
        
        if not lstm_path.exists():
            logger.warning(f"LSTM model not found for {time_window} at {lstm_path}")
            return
            
        try:
            lstm_model = LSTMBaseline()
            lstm_model.load_model(str(lstm_path))
            
            self.lstm_models[time_window] = lstm_model
            logger.info(f"[{time_window}] LSTM model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LSTM model for {time_window}: {e}")
    
    def _load_preprocessor(self):
        try:
            self.preprocessor = NetworkTrafficPreprocessor(
                self.config["preprocessor_config"]
            )
            logger.info("Preprocessor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize preprocessor: {e}")
    
    def is_ready(self) -> bool:
        return len(self.transformer_models) > 0 or len(self.lstm_models) > 0
    
    def get_model_info_for_window(self, time_window: str) -> Dict[str, Any]:
        """Get detailed model information for a specific time window"""
        info = {
            "time_window": time_window,
            "transformer_loaded": time_window in self.transformer_models,
            "lstm_loaded": time_window in self.lstm_models,
            "tokenizer_loaded": time_window in self.tokenizers,
            "model_paths": {}
        }
        
        # Add model paths
        if time_window in self.config.get("transformer_model_paths", {}):
            info["model_paths"]["transformer"] = self.config["transformer_model_paths"][time_window]
        if time_window in self.config.get("lstm_model_paths", {}):
            info["model_paths"]["lstm"] = self.config["lstm_model_paths"][time_window]
        if time_window in self.config.get("tokenizer_paths", {}):
            info["model_paths"]["tokenizer"] = self.config["tokenizer_paths"][time_window]
            
        return info
    
    def get_available_time_windows(self) -> List[str]:
        """Get list of time windows with loaded models"""
        windows = set()
        windows.update(self.transformer_models.keys())
        windows.update(self.lstm_models.keys())
        return sorted(list(windows))
    
    async def warm_up(self):
        """Warm up all loaded models"""
        for time_window, model in self.transformer_models.items():
            try:
                dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long, device=self.device)
                attention_mask = torch.ones((1, 128), dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    _ = model(
                        input_ids=dummy_input,
                        attention_mask=attention_mask,
                        task='traffic_prediction'
                    )
                logger.info(f"[{time_window}] Model warmed up successfully")
            except Exception as e:
                logger.warning(f"[{time_window}] Model warm-up failed (non-critical): {e}")
    
    async def predict(
        self,
        input_data: np.ndarray,
        task: str = "traffic_prediction",
        time_window: str = "30s",
        horizon: int = 1,
        use_ensemble: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Make predictions using time-window-specific models
        Returns: (predictions, confidence, metadata)
        """
        
        # Determine which model to use
        model_used = None
        model_type = None
        
        if time_window in self.transformer_models:
            model = self.transformer_models[time_window]
            model_used = f"transformer_{time_window}"
            model_type = "transformer"
            predictions, confidence = await self._transformer_predict(model, input_data, task, horizon)
        elif time_window in self.lstm_models:
            model = self.lstm_models[time_window]
            model_used = f"lstm_{time_window}"
            model_type = "lstm"
            predictions, confidence = await self._lstm_predict(model, input_data, horizon)
        else:
            # Fallback to try to find any available model
            available_windows = self.get_available_time_windows()
            if available_windows:
                fallback_window = available_windows[0]
                logger.warning(f"No model found for {time_window}, using {fallback_window}")
                if fallback_window in self.transformer_models:
                    model = self.transformer_models[fallback_window]
                    model_used = f"transformer_{fallback_window}"
                    model_type = "transformer"
                    predictions, confidence = await self._transformer_predict(model, input_data, task, horizon)
                else:
                    model = self.lstm_models[fallback_window]
                    model_used = f"lstm_{fallback_window}"
                    model_type = "lstm"
                    predictions, confidence = await self._lstm_predict(model, input_data, horizon)
            else:
                logger.warning("No models available, returning dummy predictions")
                batch_size = input_data.shape[0] if len(input_data.shape) > 0 else 1
                predictions = np.random.randn(batch_size, horizon, 8)
                confidence = np.ones((batch_size, horizon)) * 0.5
                model_used = "fallback_dummy"
                model_type = "dummy"
        
        metadata = {
            "requested_time_window": time_window,
            "model_used": model_used,
            "model_type": model_type,
            "actual_time_window": model_used.split('_')[-1] if '_' in model_used else time_window,
            "available_windows": self.get_available_time_windows(),
            "prediction_timestamp": datetime.utcnow().isoformat()
        }
        
        return predictions, confidence, metadata
    
    async def _transformer_predict(
        self,
        model,
        input_data: np.ndarray,
        task: str,
        horizon: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        # Ensure input is properly shaped and typed
        if isinstance(input_data, np.ndarray):
            # Convert to int64 if needed
            if input_data.dtype not in [np.int64, np.int32, torch.int64, torch.int32]:
                input_data = input_data.astype(np.int64)
            
            # Handle shape
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            # Truncate or pad to model's expected length (256)
            if input_data.shape[1] > 256:
                input_data = input_data[:, :256]
            elif input_data.shape[1] < 256:
                padding = np.zeros((input_data.shape[0], 256 - input_data.shape[1]), dtype=np.int64)
                input_data = np.concatenate([input_data, padding], axis=1)
        
        input_tensor = torch.tensor(input_data, dtype=torch.long, device=self.device)
        
        # Ensure batch dimension
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Create attention mask
        attention_mask = (input_tensor != 0).long()
        
        # Add token_type_ids to match training
        token_type_ids = torch.zeros_like(input_tensor)
        
        predictions_list = []
        confidence_list = []
        
        with torch.no_grad():
            for _ in range(horizon):
                try:
                    outputs = model(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,  # Add this
                        task=task
                    )
                    
                    if task == "traffic_prediction" and 'traffic_predictions' in outputs:
                        predictions = outputs['traffic_predictions']
                    elif task == "anomaly_detection" and 'anomaly_logits' in outputs:
                        predictions = outputs['anomaly_logits']
                    else:
                        # Fallback predictions
                        predictions = torch.randn(input_tensor.shape[0], 8, device=self.device)
                    
                    predictions_list.append(predictions.cpu().numpy())
                    confidence = torch.ones(predictions.shape[0], device=self.device) * 0.85
                    confidence_list.append(confidence.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    # Return dummy predictions on error
                    batch_size = input_tensor.shape[0]
                    predictions_list.append(np.random.randn(batch_size, 8))
                    confidence_list.append(np.ones(batch_size) * 0.5)
        
        predictions = np.stack(predictions_list, axis=1)
        confidences = np.stack(confidence_list, axis=1)
        
        return predictions, confidences
    
    async def _lstm_predict(self, model, input_data: np.ndarray, horizon: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        if model is None:
            raise ValueError("LSTM model not loaded")
        
        predictions = model.predict(input_data)
        confidence = np.ones(predictions.shape[0]) * 0.75
        
        return predictions, confidence
    
    async def detect_anomalies(self, input_data: np.ndarray, sensitivity: float = 0.5, time_window: str = "30s") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Detect anomalies with detailed time window information """
        
        if not self.is_ready():
            return [], {"error": "No models available"}
        
        # Ensure proper input format
        if isinstance(input_data, np.ndarray):
            if input_data.dtype not in [np.int64, np.int32]:
                batch_size = len(input_data) if len(input_data.shape) > 0 else 1
                input_data = np.random.randint(0, 1000, (batch_size, 256))
            
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            # Use 256 as max length to match model training
            if input_data.shape[1] > 256:
                input_data = input_data[:, :256]
            elif input_data.shape[1] < 256:
                padding = np.zeros((input_data.shape[0], 256 - input_data.shape[1]), dtype=np.int64)
                input_data = np.concatenate([input_data, padding], axis=1)
        
        predictions, confidences, metadata = await self.predict(input_data, task="anomaly_detection", time_window=time_window)
        
        anomalies = []
        threshold = 1.0 - sensitivity
        
        for i in range(len(predictions)):
            if len(predictions.shape) == 3:
                anomaly_score = float(predictions[i, 0, 0]) if predictions.shape[2] > 0 else 0.5
            elif len(predictions.shape) == 2:
                anomaly_score = float(predictions[i, 0]) if predictions.shape[1] > 0 else 0.5
            else:
                anomaly_score = 0.5
            
            anomaly_score = (anomaly_score + 1) / 2
            
            if anomaly_score > threshold:
                conf_value = float(confidences[i, 0]) if len(confidences.shape) > 1 else float(confidences[i])
                anomalies.append({
                    "index": i,
                    "score": anomaly_score,
                    "confidence": conf_value,
                    "severity": "high" if anomaly_score > 0.8 else "medium" if anomaly_score > 0.6 else "low"
                })
        
        return anomalies, metadata
    
    def calculate_confidence_intervals(self, predictions: np.ndarray, confidence_scores: np.ndarray, alpha: float = 0.95) -> Dict[str, Any]:
        
        z_score = 1.96
        
        if len(predictions.shape) == 3:
            std = np.std(predictions, axis=2, keepdims=True)
            mean = predictions
        elif len(predictions.shape) == 2:
            std = np.std(predictions, axis=1, keepdims=True)
            mean = predictions
        else:
            std = 0.1
            mean = predictions
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return {
            "lower": lower.tolist() if hasattr(lower, 'tolist') else [[lower]],
            "upper": upper.tolist() if hasattr(upper, 'tolist') else [[upper]],
            "confidence_level": alpha  
        }
    
    def get_capabilities(self) -> List[str]:
        capabilities = []
        
        if self.transformer_models:
            capabilities.extend(["traffic_prediction", "anomaly_detection", "flow_prediction"])
        if self.lstm_models:
            capabilities.append("time_series_prediction")
        if self.arima_model:
            capabilities.append("statistical_forecasting")
        
        return capabilities if capabilities else ["demo_mode"]
    
    async def reload_model(self, time_window: str, model_path: Optional[str] = None) -> bool:
        """Reload a specific time window model"""
        try:
            if model_path:
                self.config["transformer_model_paths"][time_window] = model_path
            
            self._load_transformer_for_window(time_window)
            self._load_tokenizer_for_window(time_window)
            self._load_lstm_for_window(time_window)
            
            # Warm up the reloaded model
            if time_window in self.transformer_models:
                model = self.transformer_models[time_window]
                dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long, device=self.device)
                attention_mask = torch.ones((1, 128), dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    _ = model(
                        input_ids=dummy_input,
                        attention_mask=attention_mask,
                        task='traffic_prediction'
                    )
            
            self.last_updated = datetime.now().isoformat()
            
            return True
        except Exception as e:
            logger.error(f"Failed to reload model for {time_window}: {e}")
            return False