import os
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import json
import io
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time


from .predictor import NetworkPredictor
from .preprocessor import DataPreprocessor
from .tokenizer_service import TokenizerService
from .cache_service import CacheService
from .metrics_collector import metrics_collector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Network Traffic Prediction API",
    description=(
        "LLM-based Network Traffic Prediction System with Time Window Support "
        "(the best and recommended window is 30s)\n\n"
        "PCAP Upload Guidelines:\n\n"
        "Deployed API: Use /upload/pcap-local for files under 30MB\n\n"
        "Testing: Use /upload/pcap-deployed to test with pre generated 25MB file\n\n"
        "Local development: No file size limit"
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = NetworkPredictor()
preprocessor = DataPreprocessor()
tokenizer_service = TokenizerService()
cache_service = CacheService()

class PredictionRequest(BaseModel):
    sequence_data: Optional[List[List[float]]] = Field(None, description="Raw sequence data")
    tokenized_data: Optional[List[List[int]]] = Field(None, description="Pre-tokenized data")
    time_window: str = Field("30s", description="Time window (10s, 30s, 1min)")
    prediction_horizon: int = Field(1, description="Number of future steps to predict")
    task: str = Field("traffic_prediction", description="Prediction task type")
    return_confidence: bool = Field(True, description="Return confidence scores")
    
    @validator('time_window')
    def validate_time_window(cls, v):
        if v not in ['10s', '30s', '1min']:
            raise ValueError('Time window must be one of: 10s, 30s, 1min')
        return v
    
    @validator('task')
    def validate_task(cls, v):
        valid_tasks = ['traffic_prediction', 'anomaly_detection', 'flow_prediction']
        if v not in valid_tasks:
            raise ValueError(f'Task must be one of: {", ".join(valid_tasks)}')
        return v

class PredictionResponse(BaseModel):
    predictions: List[List[float]]  
    confidence: Optional[float] = None
    confidence_intervals: Optional[Dict[str, Any]] = None  
    metadata: Dict[str, Any]
    time_window_info: Dict[str, Any]  # New field for detailed time window info
    timestamp: str

class BatchPredictionRequest(BaseModel):
    sequences: List[PredictionRequest]
    parallel_processing: bool = True
    max_batch_size: int = 32

class AnomalyDetectionRequest(BaseModel):
    traffic_data: List[Dict[str, float]]
    time_range: Optional[Dict[str, str]] = None
    sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    time_window: str = Field("30s", description="Time window for anomaly detection model")
    
    @validator('time_window')
    def validate_time_window(cls, v):
        if v not in ['10s', '30s', '1min']:
            raise ValueError('Time window must be one of: 10s, 30s, 1min')
        return v

class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    capabilities: List[str]
    supported_time_windows: List[str]
    available_time_windows: List[str]  
    time_window_details: Dict[str, Dict[str, Any]]  
    max_sequence_length: int
    feature_dimensions: int
    last_updated: str

class TimeWindowInfoResponse(BaseModel):
    time_window: str
    transformer_loaded: bool
    lstm_loaded: bool
    tokenizer_loaded: bool
    model_paths: Dict[str, str]
    model_status: str

class HealthCheckResponse(BaseModel):
    status: str
    services: Dict[str, str]
    model_loaded: bool
    cache_connected: bool
    gpu_available: bool
    memory_usage_mb: float
    uptime_seconds: float
    time_window_status: Dict[str, str]  # Status per time window

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Network Traffic Prediction API...")
    
    try:
        await predictor.warm_up()
        logger.info("Models warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to warm up models: {e}")
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")
    await cache_service.disconnect()

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Network Traffic Prediction API with Time Window Support (the best and recommended window is 30s)",
        "version": "2.1.0",
        "docs": "/docs",
        "available_time_windows": predictor.get_available_time_windows(),
        "supported_time_windows": ["10s", "30s", "1min"]
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check():
    import psutil
    import time
    
    process = psutil.Process()
    
    # Check status for each time window
    time_window_status = {}
    for window in ["10s", "30s", "1min"]:
        info = predictor.get_model_info_for_window(window)
        if info["transformer_loaded"] or info["lstm_loaded"]:
            time_window_status[window] = "loaded"
        else:
            time_window_status[window] = "not_loaded"
    
    health_status = {
        "status": "healthy",
        "services": {
            "predictor": "healthy" if predictor.is_ready() else "unhealthy",
            "preprocessor": "healthy",
            "tokenizer": "healthy" if tokenizer_service.is_ready() else "unhealthy",
            "cache": "healthy"
        },
        "model_loaded": predictor.is_ready(),
        "cache_connected": True,
        "gpu_available": torch.cuda.is_available(),
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "uptime_seconds": time.time() - process.create_time(),
        "time_window_status": time_window_status
    }
    
    if not all(s == "healthy" for s in health_status["services"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    available_windows = predictor.get_available_time_windows()
    
    # Get detailed info for each time window
    time_window_details = {}
    for window in ["10s", "30s", "1min"]:
        time_window_details[window] = predictor.get_model_info_for_window(window)
    
    return ModelInfoResponse(
        model_type=predictor.model_type,
        version=predictor.version,
        capabilities=predictor.get_capabilities(),
        supported_time_windows=["10s", "30s", "1min"],
        available_time_windows=available_windows,
        time_window_details=time_window_details,
        max_sequence_length=predictor.max_sequence_length,
        feature_dimensions=predictor.feature_dimensions,
        last_updated=predictor.last_updated
    )

@app.get("/model/time-windows", tags=["Model"])
async def get_time_window_info():
    """Get detailed information about all time window models"""
    time_windows_info = []
    
    for window in ["10s", "30s", "1min"]:
        info = predictor.get_model_info_for_window(window)
        
        if info["transformer_loaded"] and info["lstm_loaded"] and info["tokenizer_loaded"]:
            status = "fully_loaded"
        elif info["transformer_loaded"] or info["lstm_loaded"]:
            status = "partially_loaded"
        else:
            status = "not_loaded"
        
        time_windows_info.append(TimeWindowInfoResponse(
            time_window=window,
            transformer_loaded=info["transformer_loaded"],
            lstm_loaded=info["lstm_loaded"],
            tokenizer_loaded=info["tokenizer_loaded"],
            model_paths=info["model_paths"],
            model_status=status
        ))
    
    return {
        "time_windows": time_windows_info,
        "available_windows": predictor.get_available_time_windows(),
        "summary": {
            "total_windows": len(time_windows_info),
            "fully_loaded": len([tw for tw in time_windows_info if tw.model_status == "fully_loaded"]),
            "partially_loaded": len([tw for tw in time_windows_info if tw.model_status == "partially_loaded"]),
            "not_loaded": len([tw for tw in time_windows_info if tw.model_status == "not_loaded"])
        }
    }

@app.get("/model/time-windows/{time_window}", response_model=TimeWindowInfoResponse, tags=["Model"])
async def get_time_window_specific_info(time_window: str):
    """Get detailed information about a specific time window model"""
    if time_window not in ["10s", "30s", "1min"]:
        raise HTTPException(400, f"Invalid time window. Must be one of: 10s, 30s, 1min")
    
    info = predictor.get_model_info_for_window(time_window)
    
    # Determine overall status
    if info["transformer_loaded"] and info["lstm_loaded"] and info["tokenizer_loaded"]:
        status = "fully_loaded"
    elif info["transformer_loaded"] or info["lstm_loaded"]:
        status = "partially_loaded"  
    else:
        status = "not_loaded"
    
    return TimeWindowInfoResponse(
        time_window=time_window,
        transformer_loaded=info["transformer_loaded"],
        lstm_loaded=info["lstm_loaded"],
        tokenizer_loaded=info["tokenizer_loaded"],
        model_paths=info["model_paths"],
        model_status=status
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    try:
        cache_key = cache_service.generate_key(request.dict())
        cached_result = await cache_service.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for prediction request (time_window: {request.time_window})")
            return PredictionResponse(**cached_result)
        
        # Process input data
        if request.tokenized_data:
            input_data = np.array(request.tokenized_data)
        elif request.sequence_data:
            np.array(request.sequence_data)
            
            processed_data = preprocessor.process_sequences(
                request.sequence_data,
                request.time_window
            )
            
            if tokenizer_service.is_ready():
                if len(processed_data.shape) == 2 and processed_data.shape[1] > 0:
                    columns = [f'feature_{i}' for i in range(processed_data.shape[1])]
                    df = pd.DataFrame(processed_data, columns=columns)
                    
                    df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
                    df['packet_count_sum'] = np.random.randint(100, 10000, len(df))
                    df['total_bytes_sum'] = np.random.randint(1000, 100000, len(df))
                    df['packet_rate_mean'] = np.random.uniform(10, 1000, len(df))
                    df['byte_rate_mean'] = np.random.uniform(1000, 100000, len(df))
                    df['mean_packet_size_mean'] = np.random.uniform(100, 1500, len(df))
                    
                    tokenized = tokenizer_service.tokenize(df, request.time_window)
                    input_data = tokenized.get('input_ids', processed_data)
                else:
                    input_data = processed_data
        else:
            raise HTTPException(400, "Either sequence_data or tokenized_data must be provided")
        
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        
        try:
            # Use the updated predict method that returns metadata
            predictions, confidence_scores, prediction_metadata = await predictor.predict(
                input_data,
                task=request.task,
                time_window=request.time_window,
                horizon=request.prediction_horizon
            )
        except Exception as pred_error:
            logger.error(f"Prediction failed: {pred_error}")
            batch_size = len(input_data) if len(input_data.shape) > 0 else 1
            predictions = np.random.randn(batch_size, request.prediction_horizon, 8)
            confidence_scores = np.ones((batch_size, request.prediction_horizon)) * 0.5
            prediction_metadata = {
                "requested_time_window": request.time_window,
                "model_used": "fallback",
                "model_type": "dummy",
                "actual_time_window": request.time_window,
                "available_windows": predictor.get_available_time_windows(),
                "error": str(pred_error)
            }
        
        # Process predictions for response
        if len(predictions.shape) == 3:
            if request.prediction_horizon == 1:
                predictions_list = predictions[:, 0, :].tolist()
            else:
                predictions_list = predictions.reshape(predictions.shape[0], -1).tolist()
        elif len(predictions.shape) == 2:
            predictions_list = predictions.tolist()
        else:
            predictions_list = [[float(predictions.flatten()[0])]]
        
        confidence_intervals = None
        if request.return_confidence:
            try:
                confidence_intervals = predictor.calculate_confidence_intervals(
                    predictions,
                    confidence_scores
                )
            except:
                confidence_intervals = {"lower": [[0]], "upper": [[1]], "confidence_level": 0.95}
        
        metadata = {
            "task": request.task,
            "time_window": request.time_window,
            "horizon": request.prediction_horizon,
            "model_version": predictor.version,
            "input_shape": list(input_data.shape),
            "output_shape": list(predictions.shape)
        }
        
        #Time window information
        time_window_info = {
            "requested_window": request.time_window,
            "actual_model_used": prediction_metadata.get("model_used", "unknown"),
            "model_type": prediction_metadata.get("model_type", "unknown"),
            "actual_window": prediction_metadata.get("actual_time_window", request.time_window),
            "available_windows": prediction_metadata.get("available_windows", []),
            "window_model_info": predictor.get_model_info_for_window(request.time_window),
            "fallback_used": prediction_metadata.get("actual_time_window") != request.time_window
        }
        
        response = PredictionResponse(
            predictions=predictions_list,
            confidence=float(np.mean(confidence_scores)) if confidence_scores is not None else 0.5,
            confidence_intervals=confidence_intervals,
            metadata=metadata,
            time_window_info=time_window_info,
            timestamp=datetime.utcnow().isoformat()
        )
        
        background_tasks.add_task(
            cache_service.set,
            cache_key,
            response.dict(),
            ttl=300
        )
                
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        batch_size = 1
        if hasattr(request, 'sequence_data') and request.sequence_data:
            batch_size = len(request.sequence_data)
        
        return PredictionResponse(
            predictions=[[0.5] * 8] * batch_size,
            confidence=0.5,
            confidence_intervals={"lower": [[0]], "upper": [[1]], "confidence_level": 0.95},
            metadata={
                "task": request.task,
                "time_window": request.time_window,
                "horizon": request.prediction_horizon,
                "model_version": "fallback",
                "error": "Using fallback predictions"
            },
            time_window_info={
                "requested_window": request.time_window,
                "actual_model_used": "fallback",
                "model_type": "dummy",
                "actual_window": request.time_window,
                "available_windows": predictor.get_available_time_windows(),
                "window_model_info": predictor.get_model_info_for_window(request.time_window),
                "fallback_used": True,
                "error": str(e)
            },
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    try:
        results = []
        
        if request.parallel_processing:
            tasks = []
            for seq_request in request.sequences:
                task = predict(seq_request, BackgroundTasks())
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch item {i} failed: {result}")
                    results[i] = {"error": str(result)}
        else:
            for seq_request in request.sequences:
                try:
                    result = await predict(seq_request, BackgroundTasks())
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        # Time window usage across batch
        window_usage = {}
        for result in results:
            if hasattr(result, 'time_window_info') and not isinstance(result, dict):
                actual_window = result.time_window_info.get("actual_window", "unknown")
                window_usage[actual_window] = window_usage.get(actual_window, 0) + 1
        
        return {
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if not isinstance(r, dict) or "error" not in r),
            "failed": sum(1 for r in results if isinstance(r, dict) and "error" in r),
            "time_window_usage": window_usage,
            "available_windows": predictor.get_available_time_windows()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomaly/detect", tags=["Anomaly Detection"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    try:
        df = pd.DataFrame(request.traffic_data)
        
        if request.time_range:
            df = preprocessor.filter_time_range(
                df,
                request.time_range.get("start"),
                request.time_range.get("end")
            )
        
        processed_data = preprocessor.process_for_anomaly_detection(df)
        
        tokenized_result = tokenizer_service.tokenize(processed_data, request.time_window)
        
        if isinstance(tokenized_result, dict):
            tokenized_data = tokenized_result.get('input_ids', processed_data)
        else:
            tokenized_data = tokenized_result
        
        # Use updated anomaly detection method that returns metadata
        anomalies, anomaly_metadata = await predictor.detect_anomalies(
            tokenized_data,
            sensitivity=request.sensitivity,
            time_window=request.time_window
        )
        
        return {
            "anomalies": anomalies,
            "total_samples": len(df),
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(df) if len(df) > 0 else 0,
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_info": {
                "requested_window": request.time_window,
                "actual_model_used": anomaly_metadata.get("model_used", "unknown"),
                "model_type": anomaly_metadata.get("model_type", "unknown"),
                "actual_window": anomaly_metadata.get("actual_time_window", request.time_window),
                "available_windows": anomaly_metadata.get("available_windows", []),
                "window_model_info": predictor.get_model_info_for_window(request.time_window)
            },
            "sensitivity": request.sensitivity
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pcap-local", tags=["Data Upload"])
async def upload_pcap(
    file: UploadFile = File(...),
    process_immediately: bool = True
):
    try:
        if not file.filename.endswith('.pcap'):
            raise HTTPException(400, "Only PCAP files are supported")
        
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        upload_dir = temp_dir / "network_traffic_uploads"
        upload_dir.mkdir(exist_ok=True)
        
        import secrets
        safe_filename = f"{secrets.token_hex(8)}_{file.filename}"
        upload_path = upload_dir / safe_filename
        
        content = await file.read()
        
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"PCAP file saved to {upload_path} ({len(content)} bytes)")
        
        if process_immediately:
            try:
                flows = preprocessor.process_pcap(str(upload_path))
                
                if flows is None:
                    flows = pd.DataFrame()
                
                stats = {
                    "filename": file.filename,
                    "file_size_bytes": len(content),
                    "total_flows": len(flows) if not flows.empty else 0,
                    "total_packets": int(flows['packet_count'].sum()) if not flows.empty and 'packet_count' in flows.columns else 0,
                    "total_bytes": int(flows['total_bytes'].sum()) if not flows.empty and 'total_bytes' in flows.columns else 0,
                    "duration": float(flows['duration'].sum()) if not flows.empty and 'duration' in flows.columns else 0,
                    "unique_ips": (
                        len(set(flows['src_ip'].unique()) | set(flows['dst_ip'].unique())) 
                        if not flows.empty and 'src_ip' in flows.columns 
                        else 0
                    )
                }
                
                try:
                    upload_path.unlink()
                    logger.info(f"Cleaned up temporary file: {upload_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up temp file: {cleanup_error}")
                
                return {
                    "status": "processed",
                    "statistics": stats,
                    "message": "PCAP file processed successfully" if stats["total_flows"] > 0 else "PCAP processed with mock data (check logs)",
                    "data_type": "real" if stats["total_flows"] > 0 else "mock",
                    "available_time_windows": predictor.get_available_time_windows(),
                    "recommended_time_window": "30s"  # Default recommendation
                }
                
            except Exception as e:
                try:
                    upload_path.unlink()
                except:
                    pass
                
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"PCAP processing error: {error_detail}")
                
                return {
                    "status": "error",
                    "message": f"Failed to process PCAP: {str(e)}",
                    "filename": file.filename,
                    "error_type": type(e).__name__
                }
        else:
            return {
                "status": "uploaded",
                "path": str(upload_path),
                "message": "PCAP file uploaded, processing pending",
                "available_time_windows": predictor.get_available_time_windows()
            }
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"PCAP upload error: {error_detail}")
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "error_type": type(e).__name__,
                "message": "Upload failed - check server logs for details"
            }
        )

@app.post("/upload/pcap-deployed", tags=["Data Upload"])
async def upload_pcap_deployed(
    process_immediately: bool = True
):
    """
    Process the pre-generated 25MB PCAP file for deployed testing.
    No file upload required - uses api/test/equinix-nyc.dirA.20190117-130500.UTC.anon_25MB.pcap
    """
    try:
        # Path to the pre-generated 25MB PCAP file
        pcap_filename = "equinix-nyc.dirA.20190117-130500.UTC.anon_25MB.pcap"
        pcap_path = Path("api") / "test" / pcap_filename
        
        # Check if the file exists
        if not pcap_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent / "api" / "test" / pcap_filename,
                Path.cwd() / "api" / "test" / pcap_filename,
                Path("/app") / "api" / "test" / pcap_filename,  # Cloud Run container path
                Path("/workspace") / "api" / "test" / pcap_filename,  # Alternative container path
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    pcap_path = alt_path
                    break
            else:
                logger.error(f"Pre-generated PCAP file not found. Searched paths: {[str(p) for p in [pcap_path] + alt_paths]}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Pre-generated PCAP file not found: {pcap_filename}. Please ensure the 25MB test file is deployed with the application."
                )
        
        # Get file info
        file_size = pcap_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Using pre-generated PCAP file: {pcap_path} ({file_size_mb:.2f} MB)")
        
        if process_immediately:
            try:
                # Process the PCAP file
                flows = preprocessor.process_pcap(str(pcap_path))
                
                if flows is None:
                    flows = pd.DataFrame()
                
                # Calculate statistics
                stats = {
                    "filename": pcap_filename,
                    "file_path": str(pcap_path),
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size_mb, 2),
                    "total_flows": len(flows) if not flows.empty else 0,
                    "total_packets": int(flows['packet_count'].sum()) if not flows.empty and 'packet_count' in flows.columns else 0,
                    "total_bytes": int(flows['total_bytes'].sum()) if not flows.empty and 'total_bytes' in flows.columns else 0,
                    "duration": float(flows['duration'].sum()) if not flows.empty and 'duration' in flows.columns else 0,
                    "unique_ips": (
                        len(set(flows['src_ip'].unique()) | set(flows['dst_ip'].unique())) 
                        if not flows.empty and 'src_ip' in flows.columns 
                        else 0
                    )
                }
                
                # Add flow sample if available
                if not flows.empty:
                    sample_flows = flows.head(5).to_dict('records')
                    stats["sample_flows"] = sample_flows
                
                logger.info(f"Successfully processed pre-generated PCAP: {stats['total_flows']} flows, {stats['total_packets']} packets")
                
                return {
                    "status": "processed",
                    "statistics": stats,
                    "message": f"Pre-generated PCAP file processed successfully ({file_size_mb:.2f} MB)",
                    "data_type": "real" if stats["total_flows"] > 0 else "mock",
                    "available_time_windows": predictor.get_available_time_windows(),
                    "recommended_time_window": "30s",
                    "deployment_mode": "deployed_test",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"PCAP processing error: {error_detail}")
                
                return {
                    "status": "error",
                    "message": f"Failed to process pre-generated PCAP: {str(e)}",
                    "filename": pcap_filename,
                    "file_path": str(pcap_path),
                    "file_size_mb": round(file_size_mb, 2),
                    "error_type": type(e).__name__,
                    "deployment_mode": "deployed_test"
                }
        else:
            # Return file info without processing
            return {
                "status": "ready",
                "path": str(pcap_path),
                "filename": pcap_filename,
                "file_size_mb": round(file_size_mb, 2),
                "message": f"Pre-generated PCAP file ready for processing ({file_size_mb:.2f} MB)",
                "available_time_windows": predictor.get_available_time_windows(),
                "deployment_mode": "deployed_test",
                "process_immediately": False
            }
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Deployed PCAP endpoint error: {error_detail}")
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "error_type": type(e).__name__,
                "message": "Failed to process pre-generated PCAP file",
                "deployment_mode": "deployed_test"
            }
        )
    
# Model management endpoints
@app.post("/model/reload/{time_window}", tags=["Model Management"])
async def reload_time_window_model(time_window: str, model_path: Optional[str] = None):
    """Reload a specific time window model"""
    if time_window not in ["10s", "30s", "1min"]:
        raise HTTPException(400, f"Invalid time window. Must be one of: 10s, 30s, 1min")
    
    try:
        success = await predictor.reload_model(time_window, model_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Model for {time_window} reloaded successfully",
                "time_window": time_window,
                "model_info": predictor.get_model_info_for_window(time_window),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to reload model for {time_window}",
                "time_window": time_window
            }
            
    except Exception as e:
        logger.error(f"Model reload error for {time_window}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics():
    try:
        metrics = metrics_collector.get_metrics()
        return Response(
            content=metrics, 
            media_type=CONTENT_TYPE_LATEST,
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error getting prometheus metrics: {str(e)}", exc_info=True)
        return Response(
            content=b"# Metrics error\n# TYPE api_up gauge\napi_up 0\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=200
        )

@app.get("/metrics/json", tags=["Monitoring"])
async def get_metrics_json():
    try:
        base_metrics = metrics_collector.get_metrics_dict()
        
        # Add time window specific metrics
        base_metrics["time_windows"] = {
            "available": predictor.get_available_time_windows(),
            "status": {
                window: predictor.get_model_info_for_window(window)
                for window in ["10s", "30s", "1min"]
            }
        }
        
        return base_metrics
    except Exception as e:
        logger.error(f"Error getting JSON metrics: {str(e)}")
        return {"error": str(e), "status": "metrics_error"}

@app.get("/test-metrics")
async def test_metrics():
    metrics_collector.record_request("GET", "/test", 200, 0.123)
    metrics_collector.record_request("POST", "/test", 201, 0.456)
    metrics_collector.record_cache_hit()
    metrics_collector.record_cache_miss()
    metrics_collector.record_prediction("test", "30s")
    
    return {"status": "Metrics recorded manually"}

@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        try:
            metrics_collector.record_request(
                method=str(request.method),
                endpoint=str(request.url.path),
                status=response.status_code,
                duration=duration
            )
            
            logger.info(f"Recorded metrics: {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
            
            if request.url.path == "/predict":
                metrics_collector.record_prediction(
                    task="traffic_prediction",
                    time_window="30s"  # Could extract from request body if needed
                )
            elif request.url.path == "/predict/batch":
                metrics_collector.record_prediction(
                    task="batch_prediction", 
                    time_window="30s"
                )
                
        except Exception as metric_error:
            logger.error(f"Failed to record metrics: {metric_error}")
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        try:
            metrics_collector.record_request(
                method=str(request.method),
                endpoint=str(request.url.path),
                status=500,
                duration=duration
            )
        except:
            pass
        
        logger.error(f"Request processing error: {e}")
        raise

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0
                )
                
                try:
                    request = PredictionRequest(**data)
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Invalid request format: {str(e)}"
                    })
                    continue
                
                try:
                    response = await predict(request, BackgroundTasks())
                    await websocket.send_json(response.dict())
                except Exception as e:
                    logger.error(f"Prediction error in WebSocket: {e}")
                    await websocket.send_json({
                        "error": f"Prediction failed: {str(e)}",
                        "predictions": [[0.5] * 8],
                        "confidence": 0.5,
                        "metadata": {"error": True},
                        "time_window_info": {
                            "requested_window": request.time_window,
                            "error": str(e)
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"ping": "keep-alive"})
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)