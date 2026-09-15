import asyncio
import os
import time
import json
import httpx
import numpy as np
import pandas as pd
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Tuple
import base64

class APIPerformanceTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.time_windows = ["10s", "30s", "1min"]
        self.results = {
            'predict': [],
            'predict_batch': [],
            'anomaly_detect': [],
            'ws_predict': [],
            'pcap_upload': [],
            'time_window_info': [],
            'model_reload': []
        }
        
    async def warmup(self, iterations):
        """Warming up the API with requests for each time window"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First check which time windows are available
            try:
                response = await client.get(f"{self.base_url}/model/time-windows")
                if response.status_code == 200:
                    data = response.json()
                    available_windows = data.get("available_windows", self.time_windows)
                    print(f"Available time windows: {available_windows}")
            except Exception as e:
                print(f"Could not fetch time window info: {e}")
            
            for i in range(iterations):
                try:
                    # Health check
                    await client.get(f"{self.base_url}/health")
                    
                    # Warmup prediction endpoint for each time window
                    for time_window in self.time_windows:
                        test_data = {
                            "sequence_data": [[np.random.randn() for _ in range(8)] for _ in range(10)],
                            "time_window": time_window,
                            "prediction_horizon": 1
                        }
                        await client.post(f"{self.base_url}/predict", json=test_data)
                except Exception as e:
                    print(f"Warmup iteration {i} failed: {e}")
        print("Warmup complete")
    
    async def test_time_window_endpoints(self):
        """Test time window specific endpoints """
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.perf_counter()
            try:
                response = await client.get(f"{self.base_url}/model/time-windows")
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    latency = (end_time - start_time) * 1000
                    data = response.json()
                    self.results['time_window_info'].append({
                        'endpoint': '/model/time-windows',
                        'latency_ms': latency,
                        'available_windows': data.get('available_windows', []),
                        'fully_loaded': data.get('summary', {}).get('fully_loaded', 0),
                        'status_code': response.status_code
                    })
                    print(f"Time windows available: {data.get('available_windows', [])}")
            except Exception as e:
                print(f"Failed to get time window info: {e}")
            
            # Test each specific time window endpoint
            for time_window in self.time_windows:
                start_time = time.perf_counter()
                try:
                    response = await client.get(f"{self.base_url}/model/time-windows/{time_window}")
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000
                        data = response.json()
                        self.results['time_window_info'].append({
                            'endpoint': f'/model/time-windows/{time_window}',
                            'time_window': time_window,
                            'latency_ms': latency,
                            'model_status': data.get('model_status', 'unknown'),
                            'transformer_loaded': data.get('transformer_loaded', False),
                            'lstm_loaded': data.get('lstm_loaded', False),
                            'tokenizer_loaded': data.get('tokenizer_loaded', False),
                            'status_code': response.status_code
                        })
                        print(f"  {time_window}: {data.get('model_status', 'unknown')}")
                except Exception as e:
                    print(f"Failed to get info for {time_window}: {e}")
    
    async def test_predict_endpoint_with_time_windows(self, iterations):
        """Test prediction endpoint with different time windows"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(iterations):
                # Rotate through time windows
                time_window = self.time_windows[i % len(self.time_windows)]
                
                # Test data
                sequence_length = np.random.randint(5, 20)
                test_data = {
                    "sequence_data": [[np.random.randn() for _ in range(8)] 
                                    for _ in range(sequence_length)],
                    "time_window": time_window,
                    "prediction_horizon": np.random.randint(1, 5),
                    "return_confidence": True
                }
                
                start_time = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.base_url}/predict",
                        json=test_data
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000
                        response_data = response.json()
                        
                        # Extract time window info from response
                        time_window_info = response_data.get('time_window_info', {})
                        
                        self.results['predict'].append({
                            'iteration': i,
                            'latency_ms': latency,
                            'sequence_length': sequence_length,
                            'requested_time_window': time_window,
                            'actual_time_window': time_window_info.get('actual_window', time_window),
                            'model_used': time_window_info.get('actual_model_used', 'unknown'),
                            'model_type': time_window_info.get('model_type', 'unknown'),
                            'fallback_used': time_window_info.get('fallback_used', False),
                            'status_code': response.status_code,
                            'response_size': len(response.content)
                        })
                    else:
                        print(f"Error in iteration {i}: {response.status_code}")
                except Exception as e:
                    print(f"Request {i} failed: {e}")
                
                if i % 20 == 0:
                    print(f"  Completed {i}/{iterations} requests")
                    await asyncio.sleep(0.1)
    
    async def test_batch_predict_with_time_windows(self, iterations):
        """Test batch prediction with different time windows"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(iterations):
                batch_size = np.random.randint(2, 10)
                sequences = []
                
                for j in range(batch_size):
                    time_window = self.time_windows[j % len(self.time_windows)]
                    sequences.append({
                        "sequence_data": [[np.random.randn() for _ in range(8)] 
                                        for _ in range(np.random.randint(5, 15))],
                        "time_window": time_window,
                        "prediction_horizon": 1
                    })
                
                batch_data = {
                    "sequences": sequences,
                    "parallel_processing": True
                }
                
                start_time = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.base_url}/predict/batch",
                        json=batch_data
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000
                        response_data = response.json()
                        
                        self.results['predict_batch'].append({
                            'iteration': i,
                            'latency_ms': latency,
                            'batch_size': batch_size,
                            'avg_latency_per_item': latency / batch_size,
                            'time_window_usage': response_data.get('time_window_usage', {}),
                            'successful': response_data.get('successful', 0),
                            'failed': response_data.get('failed', 0),
                            'status_code': response.status_code
                        })
                except Exception as e:
                    print(f"Batch request {i} failed: {e}")
                
                if i % 10 == 0:
                    print(f"  Completed {i}/{iterations} batch requests")
    
    async def test_anomaly_with_time_windows(self, iterations):
        """Test anomaly detection with different time windows"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(iterations):
                time_window = self.time_windows[i % len(self.time_windows)]
                
                # Generate traffic data
                num_samples = np.random.randint(10, 100)
                traffic_data = []
                
                for _ in range(num_samples):
                    traffic_data.append({
                        "packet_count": float(np.random.randint(100, 10000)),
                        "total_bytes": float(np.random.randint(1000, 1000000)),
                        "packet_rate": float(np.random.uniform(10, 1000)),
                        "byte_rate": float(np.random.uniform(1000, 100000))
                    })
                
                test_data = {
                    "traffic_data": traffic_data,
                    "sensitivity": np.random.uniform(0.3, 0.7),
                    "time_window": time_window
                }
                
                start_time = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.base_url}/anomaly/detect",
                        json=test_data
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000
                        result_data = response.json()
                        time_window_info = result_data.get('time_window_info', {})
                        
                        self.results['anomaly_detect'].append({
                            'iteration': i,
                            'latency_ms': latency,
                            'num_samples': num_samples,
                            'requested_time_window': time_window,
                            'actual_time_window': time_window_info.get('actual_window', time_window),
                            'model_used': time_window_info.get('actual_model_used', 'unknown'),
                            'anomalies_found': result_data.get('anomaly_count', 0),
                            'status_code': response.status_code
                        })
                except Exception as e:
                    print(f"Anomaly request {i} failed: {e}")
    
    async def test_model_reload(self):
        """Test model reload endpoints for each time window"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            for time_window in self.time_windows:
                start_time = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.base_url}/model/reload/{time_window}",
                        headers={"X-Admin-Key": os.environ.get("ADMIN_API_KEY", "")}
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000
                        result_data = response.json()
                        
                        self.results['model_reload'].append({
                            'time_window': time_window,
                            'latency_ms': latency,
                            'status': result_data.get('status', 'unknown'),
                            'model_info': result_data.get('model_info', {}),
                            'status_code': response.status_code
                        })
                        print(f"  Model reload for {time_window}: {result_data.get('status')}")
                except Exception as e:
                    print(f"Model reload failed for {time_window}: {e}")
    
    async def test_websocket_streaming_with_time_windows(self, iterations):
        """Test WebSocket streaming with different time windows"""
        import websockets
        
        try:
            uri = f"ws://127.0.0.1:8000/ws/predict"
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                for i in range(iterations):
                    # Rotate through time windows
                    time_window = self.time_windows[i % len(self.time_windows)]
                    
                    message = {
                        "sequence_data": [[np.random.randn() for _ in range(8)] 
                                        for _ in range(10)],
                        "time_window": time_window,
                        "prediction_horizon": 1
                    }
                    
                    start_time = time.perf_counter()
                    
                    try:
                        await websocket.send(json.dumps(message))
                        
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0
                        )
                        
                        end_time = time.perf_counter()
                        latency = (end_time - start_time) * 1000
                        
                        response_data = json.loads(response)
                        if 'ping' not in response_data:
                            time_window_info = response_data.get('time_window_info', {})
                            self.results['ws_predict'].append({
                                'iteration': i,
                                'latency_ms': latency,
                                'requested_time_window': time_window,
                                'actual_time_window': time_window_info.get('actual_window', time_window),
                                'response_size': len(response)
                            })
                        
                        if i % 10 == 0:
                            print(f"  Completed {i}/{iterations} WebSocket messages")
                            
                    except asyncio.TimeoutError:
                        print(f"  WebSocket message {i} timed out")
                    except Exception as e:
                        print(f"  WebSocket message {i} failed: {e}")
                        
        except Exception as e:
            print(f"WebSocket test failed: {e}")
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics including time window analysis"""
        results_summary = {}
        
        # Predict endpoint statistics with time window analysis
        if self.results['predict']:
            df = pd.DataFrame(self.results['predict'])
            results_summary['predict'] = {
                'overall': {
                    'mean_latency_ms': df['latency_ms'].mean(),
                    'median_latency_ms': df['latency_ms'].median(),
                    'p95_latency_ms': df['latency_ms'].quantile(0.95),
                    'p99_latency_ms': df['latency_ms'].quantile(0.99),
                    'min_latency_ms': df['latency_ms'].min(),
                    'max_latency_ms': df['latency_ms'].max(),
                    'std_latency_ms': df['latency_ms'].std(),
                    'throughput_rps': 1000 / df['latency_ms'].mean()
                },
                'by_time_window': {}
            }
            
            for window in self.time_windows:
                window_df = df[df['requested_time_window'] == window]
                if not window_df.empty:
                    results_summary['predict']['by_time_window'][window] = {
                        'mean_latency_ms': window_df['latency_ms'].mean(),
                        'median_latency_ms': window_df['latency_ms'].median(),
                        'p95_latency_ms': window_df['latency_ms'].quantile(0.95),
                        'fallback_rate': window_df['fallback_used'].mean() if 'fallback_used' in window_df else 0,
                        'count': len(window_df)
                    }
            
            # Model type analysis
            if 'model_type' in df.columns:
                model_types = df['model_type'].value_counts().to_dict()
                results_summary['predict']['model_types_used'] = model_types
            
            print("\n/predict Endpoint Overall:")
            for key, value in results_summary['predict']['overall'].items():
                print(f"  {key}: {value:.2f}")
            
            print("\n/predict By Time Window:")
            for window, stats in results_summary['predict']['by_time_window'].items():
                print(f"  {window}:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")
        
        # Batch predict statistics
        if self.results['predict_batch']:
            df = pd.DataFrame(self.results['predict_batch'])
            results_summary['predict_batch'] = {
                'mean_latency_ms': df['latency_ms'].mean(),
                'median_latency_ms': df['latency_ms'].median(),
                'p95_latency_ms': df['latency_ms'].quantile(0.95),
                'p99_latency_ms': df['latency_ms'].quantile(0.99),
                'avg_per_item_ms': df['avg_latency_per_item'].mean(),
                'throughput_rps': 1000 / df['avg_latency_per_item'].mean(),
                'avg_successful': df['successful'].mean() if 'successful' in df else 0,
                'avg_failed': df['failed'].mean() if 'failed' in df else 0
            }
            
            print("\n/predict/batch Endpoint:")
            for key, value in results_summary['predict_batch'].items():
                print(f"  {key}: {value:.2f}")
        
        if self.results['anomaly_detect']:
            df = pd.DataFrame(self.results['anomaly_detect'])
            results_summary['anomaly_detect'] = {
                'overall': {
                    'mean_latency_ms': df['latency_ms'].mean(),
                    'median_latency_ms': df['latency_ms'].median(),
                    'p95_latency_ms': df['latency_ms'].quantile(0.95),
                    'p99_latency_ms': df['latency_ms'].quantile(0.99),
                    'avg_anomalies_found': df['anomalies_found'].mean(),
                    'throughput_rps': 1000 / df['latency_ms'].mean()
                },
                'by_time_window': {}
            }
            
            for window in self.time_windows:
                window_df = df[df['requested_time_window'] == window]
                if not window_df.empty:
                    results_summary['anomaly_detect']['by_time_window'][window] = {
                        'mean_latency_ms': window_df['latency_ms'].mean(),
                        'avg_anomalies': window_df['anomalies_found'].mean(),
                        'count': len(window_df)
                    }
            
            print("\n/anomaly/detect Endpoint:")
            for key, value in results_summary['anomaly_detect']['overall'].items():
                print(f"  {key}: {value:.2f}")
        
        if self.results['ws_predict']:
            df = pd.DataFrame(self.results['ws_predict'])
            results_summary['ws_predict'] = {
                'mean_latency_ms': df['latency_ms'].mean(),
                'median_latency_ms': df['latency_ms'].median(),
                'p95_latency_ms': df['latency_ms'].quantile(0.95),
                'p99_latency_ms': df['latency_ms'].quantile(0.99),
                'throughput_mps': 1000 / df['latency_ms'].mean()
            }
            
            print("\nWebSocket /ws/predict:")
            for key, value in results_summary['ws_predict'].items():
                print(f"  {key}: {value:.2f}")
        
        if self.results['time_window_info']:
            df = pd.DataFrame(self.results['time_window_info'])
            window_status = {}
            
            for window in self.time_windows:
                window_data = df[df['time_window'] == window]
                if not window_data.empty:
                    latest = window_data.iloc[-1]
                    window_status[window] = {
                        'model_status': latest.get('model_status', 'unknown'),
                        'transformer_loaded': latest.get('transformer_loaded', False),
                        'lstm_loaded': latest.get('lstm_loaded', False),
                        'tokenizer_loaded': latest.get('tokenizer_loaded', False)
                    }
            
            results_summary['time_window_status'] = window_status
            
            print("\nTime Window Status:")
            for window, status in window_status.items():
                print(f"  {window}: {status}")
        
        # Model reload statistics
        if self.results['model_reload']:
            df = pd.DataFrame(self.results['model_reload'])
            results_summary['model_reload'] = {
                'mean_reload_time_ms': df['latency_ms'].mean() if 'latency_ms' in df else 0,
                'success_rate': (df['status'] == 'success').mean() if 'status' in df else 0
            }
            
            print("\nModel Reload:")
            for key, value in results_summary['model_reload'].items():
                print(f"  {key}: {value:.2f}")
        
        # PCAP upload statistics
        if self.results['pcap_upload']:
            df = pd.DataFrame(self.results['pcap_upload'])
            results_summary['pcap_upload'] = {
                'mean_processing_time_ms': df['processing_time_ms'].mean(),
                'median_processing_time_ms': df['processing_time_ms'].median(),
                'avg_throughput_mbps': df['throughput_mbps'].mean(),
                'total_flows': df['flows_extracted'].sum(),
                'total_packets': df['packets_processed'].sum(),
                'avg_file_size_mb': df['file_size_mb'].mean()
            }
            
            print("\nPCAP Upload /upload/pcap-local:")
            for key, value in results_summary['pcap_upload'].items():
                print(f"  {key}: {value:.2f}")
        
        return results_summary
    
    def save_results(self, output_dir="api/test/results/api_performance"):
        """Save detailed results including time window analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        for endpoint, data in self.results.items():
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output_path / f"{endpoint}_results.csv", index=False)
                print(f"Saved {endpoint} results to {output_path / f'{endpoint}_results.csv'}")
        
        # Save summary
        summary = self.calculate_statistics()
        with open(output_path / "performance_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate time window comparison report
        self.generate_time_window_report(output_path, summary)
        
        return summary
    

    def generate_time_window_report(self, output_path: Path, summary: Dict):
        """Generate a comprehensive CSV report comparing time window performance"""
        
        report_data = []
        
        time_windows = getattr(self, 'time_windows', ['10s', '30s', '1min'])
        
        for window in time_windows:
            row = {
                'time_window': window,
                'generated_at': pd.Timestamp.now()
            }
            
            if 'time_window_status' in summary and window in summary['time_window_status']:
                status = summary['time_window_status'][window]
                row.update({
                    'model_status': status.get('model_status', 'unknown'),
                    'transformer_loaded': status.get('transformer_loaded', False),
                    'lstm_loaded': status.get('lstm_loaded', False),
                    'tokenizer_loaded': status.get('tokenizer_loaded', False)
                })
            else:
                row.update({
                    'model_status': 'unknown',
                    'transformer_loaded': False,
                    'lstm_loaded': False,
                    'tokenizer_loaded': False
                })
            
            # Add prediction performance metrics
            if 'predict' in summary and 'by_time_window' in summary['predict']:
                if window in summary['predict']['by_time_window']:
                    stats = summary['predict']['by_time_window'][window]
                    row.update({
                        'predict_mean_latency_ms': stats.get('mean_latency_ms', np.nan),
                        'predict_median_latency_ms': stats.get('median_latency_ms', np.nan),
                        'predict_p95_latency_ms': stats.get('p95_latency_ms', np.nan),
                        'predict_fallback_rate': stats.get('fallback_rate', 0.0),
                        'predict_request_count': stats.get('count', 0)
                    })
                else:
                    row.update({
                        'predict_mean_latency_ms': np.nan,
                        'predict_median_latency_ms': np.nan,
                        'predict_p95_latency_ms': np.nan,
                        'predict_fallback_rate': 0.0,
                        'predict_request_count': 0
                    })
            
            # Add anomaly detection performance metrics
            if 'anomaly_detect' in summary and 'by_time_window' in summary['anomaly_detect']:
                if window in summary['anomaly_detect']['by_time_window']:
                    anomaly_stats = summary['anomaly_detect']['by_time_window'][window]
                    row.update({
                        'anomaly_mean_latency_ms': anomaly_stats.get('mean_latency_ms', np.nan),
                        'anomaly_avg_anomalies_found': anomaly_stats.get('avg_anomalies', 0.0),
                        'anomaly_request_count': anomaly_stats.get('count', 0)
                    })
                else:
                    row.update({
                        'anomaly_mean_latency_ms': np.nan,
                        'anomaly_avg_anomalies_found': 0.0,
                        'anomaly_request_count': 0
                    })
            
            report_data.append(row)
        
        # Add overall statistics row
        overall_row = {
            'time_window': 'OVERALL',
            'generated_at': pd.Timestamp.now()
        }
        
        # Add overall prediction metrics
        if 'predict' in summary and 'overall' in summary['predict']:
            overall_stats = summary['predict']['overall']
            overall_row.update({
                'predict_mean_latency_ms': overall_stats.get('mean_latency_ms', np.nan),
                'predict_median_latency_ms': overall_stats.get('median_latency_ms', np.nan),
                'predict_p95_latency_ms': overall_stats.get('p95_latency_ms', np.nan),
                'predict_p99_latency_ms': overall_stats.get('p99_latency_ms', np.nan),
                'predict_min_latency_ms': overall_stats.get('min_latency_ms', np.nan),
                'predict_max_latency_ms': overall_stats.get('max_latency_ms', np.nan),
                'predict_std_latency_ms': overall_stats.get('std_latency_ms', np.nan),
                'predict_throughput_rps': overall_stats.get('throughput_rps', np.nan)
            })
        
        # Add overall anomaly metrics
        if 'anomaly_detect' in summary and 'overall' in summary['anomaly_detect']:
            anomaly_overall = summary['anomaly_detect']['overall']
            overall_row.update({
                'anomaly_mean_latency_ms': anomaly_overall.get('mean_latency_ms', np.nan),
                'anomaly_median_latency_ms': anomaly_overall.get('median_latency_ms', np.nan),
                'anomaly_p95_latency_ms': anomaly_overall.get('p95_latency_ms', np.nan),
                'anomaly_p99_latency_ms': anomaly_overall.get('p99_latency_ms', np.nan),
                'anomaly_avg_anomalies_found': anomaly_overall.get('avg_anomalies_found', 0.0),
                'anomaly_throughput_rps': anomaly_overall.get('throughput_rps', np.nan)
            })
        
        # Add batch prediction metrics
        if 'predict_batch' in summary:
            batch_stats = summary['predict_batch']
            overall_row.update({
                'batch_mean_latency_ms': batch_stats.get('mean_latency_ms', np.nan),
                'batch_median_latency_ms': batch_stats.get('median_latency_ms', np.nan),
                'batch_p95_latency_ms': batch_stats.get('p95_latency_ms', np.nan),
                'batch_avg_per_item_ms': batch_stats.get('avg_per_item_ms', np.nan),
                'batch_throughput_rps': batch_stats.get('throughput_rps', np.nan),
                'batch_avg_successful': batch_stats.get('avg_successful', 0.0),
                'batch_avg_failed': batch_stats.get('avg_failed', 0.0)
            })
        
        # Add WebSocket metrics
        if 'ws_predict' in summary:
            ws_stats = summary['ws_predict']
            overall_row.update({
                'ws_mean_latency_ms': ws_stats.get('mean_latency_ms', np.nan),
                'ws_median_latency_ms': ws_stats.get('median_latency_ms', np.nan),
                'ws_p95_latency_ms': ws_stats.get('p95_latency_ms', np.nan),
                'ws_throughput_mps': ws_stats.get('throughput_mps', np.nan)
            })
        
        # Add model reload metrics
        if 'model_reload' in summary:
            reload_stats = summary['model_reload']
            overall_row.update({
                'model_reload_mean_time_ms': reload_stats.get('mean_reload_time_ms', np.nan),
                'model_reload_success_rate': reload_stats.get('success_rate', 0.0)
            })
        
        report_data.append(overall_row)
        
        # Add model type usage statistics but filter out inconsistent data
        if 'predict' in summary and 'model_types_used' in summary['predict']:
            total_fallback_rate = 0
            if 'predict' in summary and 'by_time_window' in summary['predict']:
                window_stats = summary['predict']['by_time_window'].values()
                total_requests = sum(s.get('count', 0) for s in window_stats)
                total_fallbacks = sum(s.get('count', 0) * s.get('fallback_rate', 0) for s in window_stats)
                if total_requests > 0:
                    total_fallback_rate = total_fallbacks / total_requests
            
            for model_type, count in summary['predict']['model_types_used'].items():
                if model_type == 'dummy' and total_fallback_rate == 0:
                    continue
                
                model_status_list = [summary.get('time_window_status', {}).get(w, {}).get('model_status', 'unknown') 
                                    for w in time_windows]
                
                if model_type == 'dummy' and all(status == 'fully_loaded' for status in model_status_list):
                    continue
                
                if model_type in ['transformer', 'lstm', 'fallback'] or \
                (model_type == 'dummy' and total_fallback_rate > 0):
                    model_type_row = {
                        'time_window': f'MODEL_TYPE_{model_type}',
                        'generated_at': pd.Timestamp.now(),
                        'predict_request_count': count
                    }
                    report_data.append(model_type_row)
        
        df = pd.DataFrame(report_data)
        
        column_order = [
            'time_window', 'generated_at', 
            'model_status', 'transformer_loaded', 'lstm_loaded', 'tokenizer_loaded',
            'predict_mean_latency_ms', 'predict_median_latency_ms', 'predict_p95_latency_ms', 
            'predict_p99_latency_ms', 'predict_min_latency_ms', 'predict_max_latency_ms',
            'predict_std_latency_ms', 'predict_fallback_rate', 'predict_request_count',
            'predict_throughput_rps',
            'anomaly_mean_latency_ms', 'anomaly_median_latency_ms', 'anomaly_p95_latency_ms',
            'anomaly_p99_latency_ms', 'anomaly_avg_anomalies_found', 'anomaly_request_count',
            'anomaly_throughput_rps',
            'batch_mean_latency_ms', 'batch_median_latency_ms', 'batch_p95_latency_ms',
            'batch_avg_per_item_ms', 'batch_throughput_rps', 'batch_avg_successful', 'batch_avg_failed',
            'ws_mean_latency_ms', 'ws_median_latency_ms', 'ws_p95_latency_ms', 'ws_throughput_mps',
            'model_reload_mean_time_ms', 'model_reload_success_rate'
        ]
        
        ordered_columns = [col for col in column_order if col in df.columns]
        df = df[ordered_columns]
        
        csv_path = output_path / "time_window_performance_report.csv"
        df.to_csv(csv_path, index=False, float_format='%.2f', na_rep='N/A')
        
        print(f"Saved time window performance report to {csv_path}")
        
        # Print summary to console
        print("\nTime Window Performance Summary:")
        print("-" * 60)
        
        for _, row in df.iterrows():
            if row['time_window'] in time_windows:
                print(f"\n{row['time_window']}:")
                print(f"  Model Status: {row.get('model_status', 'N/A')}")
                print(f"  Mean Latency: {row.get('predict_mean_latency_ms', 'N/A'):.2f} ms" if not pd.isna(row.get('predict_mean_latency_ms')) else "  Mean Latency: N/A")
                print(f"  P95 Latency: {row.get('predict_p95_latency_ms', 'N/A'):.2f} ms" if not pd.isna(row.get('predict_p95_latency_ms')) else "  P95 Latency: N/A")
                print(f"  Fallback Rate: {row.get('predict_fallback_rate', 0)*100:.1f}%")
                print(f"  Request Count: {int(row.get('predict_request_count', 0))}")
        
        return df
    
    async def test_pcap_upload(self, pcap_dir="code/data/raw", num_files=5):
        """Test PCAP upload (unchanged from original)"""
        pcap_files = list(Path(pcap_dir).glob("*.pcap"))[:num_files]
        
        if not pcap_files:
            print(f"No PCAP files found in {pcap_dir}")
            return
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, pcap_file in enumerate(pcap_files):
                file_size = pcap_file.stat().st_size
                
                # Read file
                with open(pcap_file, 'rb') as f:
                    file_content = f.read()
                
                files = {
                    'file': (pcap_file.name, file_content, 'application/vnd.tcpdump.pcap')
                }
                
                start_time = time.perf_counter()
                try:
                    response = await client.post(
                        f"{self.base_url}/upload/pcap-local",
                        files=files,
                        params={'process_immediately': True}
                    )
                    end_time = time.perf_counter()
                    
                    if response.status_code == 200:
                        processing_time = (end_time - start_time) * 1000
                        result_data = response.json()
                        
                        self.results['pcap_upload'].append({
                            'file': pcap_file.name,
                            'file_size_mb': file_size / (1024 * 1024),
                            'processing_time_ms': processing_time,
                            'flows_extracted': result_data.get('statistics', {}).get('total_flows', 0),
                            'packets_processed': result_data.get('statistics', {}).get('total_packets', 0),
                            'throughput_mbps': (file_size * 8) / (processing_time * 1000),
                            'recommended_time_window': result_data.get('recommended_time_window', 'unknown')
                        })
                        print(f"  Processed {pcap_file.name}: {processing_time:.2f}ms")
                except Exception as e:
                    print(f"PCAP upload failed for {pcap_file.name}: {e}")

async def main():
    tester = APIPerformanceTester()
    
    print("Deployed API Performance Testing with Time Window Support")
    print("=" * 60)

    print("\n Warming up API ")
    await tester.warmup(iterations=10)
    
    print("\n Testing time window information endpoints ")
    await tester.test_time_window_endpoints()
    
    print("\n Testing prediction endpoints with time windows ")
    await tester.test_predict_endpoint_with_time_windows(iterations=100)
    
    print("\n Testing batch prediction with mixed time windows ")
    await tester.test_batch_predict_with_time_windows(iterations=50)
    
    print("\n Testing anomaly detection with time windows ")
    await tester.test_anomaly_with_time_windows(iterations=15)
    
    print("\n Testing WebSocket streaming with time windows ")
    await tester.test_websocket_streaming_with_time_windows(iterations=50)
    
    print("\n Testing PCAP upload ")
    await tester.test_pcap_upload(num_files=5)
    
    print("\n Testing model reload endpoints ")
    await tester.test_model_reload()  
    
    print("\n Analyzing results and generating reports ")
    tester.save_results()
    
    print("\n" + "=" * 60)
    print("Testing Complete")


if __name__ == "__main__":
    asyncio.run(main())