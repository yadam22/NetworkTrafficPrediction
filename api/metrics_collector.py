from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
import time
import psutil
import torch
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

registry = REGISTRY

try:
    from prometheus_client import REGISTRY as default_registry
    collectors = list(default_registry._collector_to_names.keys())
    for collector in collectors:
        try:
            default_registry.unregister(collector)
        except:
            pass
except:
    pass

PROMETHEUS_MULTIPROC_DIR = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
if PROMETHEUS_MULTIPROC_DIR:
    from prometheus_client import multiprocess
    try:
        import glob
        for f in glob.glob(os.path.join(PROMETHEUS_MULTIPROC_DIR, "*.db")):
            os.remove(f)
        logger.info(f"Cleaned up stale metrics in {PROMETHEUS_MULTIPROC_DIR}")
    except Exception as e:
        logger.warning(f"Could not clean metrics directory: {e}")

process = psutil.Process(os.getpid())

process_cpu_seconds_total = Gauge(
    'process_cpu_seconds_total',
    'Total user and system CPU time spent in seconds',
    registry=registry
)

process_resident_memory_bytes = Gauge(
    'process_resident_memory_bytes',
    'Resident memory size in bytes',
    registry=registry
)

process_virtual_memory_bytes = Gauge(
    'process_virtual_memory_bytes',
    'Virtual memory size in bytes',
    registry=registry
)

process_open_fds = Gauge(
    'process_open_fds',
    'Number of open file descriptors',
    registry=registry
)

process_num_threads = Gauge(
    'process_num_threads',
    'Number of threads',
    registry=registry
)

api_process_cpu_percent = Gauge(
    'api_process_cpu_percent',
    'Current CPU usage percentage of the API process',
    registry=registry
)

api_process_memory_mb = Gauge(
    'api_process_memory_mb',
    'Current memory usage in MB of the API process',
    registry=registry
)

request_count = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=registry,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

prediction_count = Counter(
    'predictions_total',
    'Total number of predictions',
    ['task', 'time_window'],
    registry=registry
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_type', 'time_window'],
    registry=registry,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

cache_hits = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    registry=registry
)

pcap_files_processed = Counter(
    'pcap_files_processed_total',
    'Total number of PCAP files processed',
    ['status'],
    registry=registry
)

anomaly_detections = Counter(
    'anomaly_detections_total',
    'Total number of anomaly detections performed',
    registry=registry
)

network_anomaly_score = Gauge(
    'network_anomaly_score',
    'Current network anomaly score',
    registry=registry
)

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System-wide CPU usage percentage',
    registry=registry
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System-wide memory usage in bytes',
    ['type'],
    registry=registry
)

model_loaded = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)',
    ['model_type'],
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_type'],
    registry=registry
)

model_loss = Gauge(
    'model_loss',
    'Current model loss',
    ['model_type'],
    registry=registry
)

network_packet_rate = Gauge(
    'network_packet_rate',
    'Current network packet rate',
    registry=registry
)

network_byte_rate = Gauge(
    'network_byte_rate',
    'Current network byte rate',
    registry=registry
)

network_flow_count = Gauge(
    'network_flow_count',
    'Current number of network flows',
    ['protocol'],
    registry=registry
)

network_traffic_volume_bytes = Gauge(
    'network_traffic_volume_bytes',
    'Total network traffic volume',
    ['direction'],
    registry=registry
)

class MetricsCollector:
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.multiprocess = bool(PROMETHEUS_MULTIPROC_DIR)
        
        self._update_process_metrics()
        self._update_system_metrics()
        
        logger.info(f"Enhanced metrics collector initialized (multiprocess={self.multiprocess})")
    
    def _update_process_metrics(self):
        try:
            cpu_times = self.process.cpu_times()
            process_cpu_seconds_total.set(cpu_times.user + cpu_times.system)
            
            cpu_percent = self.process.cpu_percent()
            api_process_cpu_percent.set(cpu_percent)
            
            memory_info = self.process.memory_info()
            process_resident_memory_bytes.set(memory_info.rss)
            process_virtual_memory_bytes.set(memory_info.vms)
            api_process_memory_mb.set(memory_info.rss / (1024 * 1024))
            
            try:
                process_open_fds.set(self.process.num_fds())
            except AttributeError:
                try:
                    process_open_fds.set(len(self.process.open_files()))
                except:
                    process_open_fds.set(0)
            
            process_num_threads.set(self.process.num_threads())
            
            logger.debug(f"Process metrics updated: CPU={cpu_percent:.2f}%, Memory={memory_info.rss/(1024*1024):.2f}MB")
            
        except Exception as e:
            logger.error(f"Error updating process metrics: {e}", exc_info=True)
    
    def _update_system_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            system_cpu_usage_percent.set(cpu_percent)
            
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.labels(type='used').set(memory.used)
            system_memory_usage_bytes.labels(type='available').set(memory.available)
            system_memory_usage_bytes.labels(type='total').set(memory.total)
            
            logger.debug(f"System metrics updated: CPU={cpu_percent:.2f}%, Memory={memory.percent:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}", exc_info=True)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        try:
            request_count.labels(
                method=method, 
                endpoint=endpoint, 
                status=str(status)
            ).inc()
            
            request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            self._update_process_metrics()
            
            logger.debug(f"Recorded request: {method} {endpoint} {status} {duration:.3f}s")
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}", exc_info=True)
    
    def record_prediction(self, task: str, time_window: str, latency: float = None):
        try:
            prediction_count.labels(task=task, time_window=time_window).inc()
            
            if latency is not None:
                prediction_latency.labels(
                    model_type=task,
                    time_window=time_window
                ).observe(latency)
            
            logger.debug(f"Recorded prediction: {task} {time_window}")
        except Exception as e:
            logger.error(f"Error recording prediction metrics: {e}")
    
    def record_cache_hit(self):
        try:
            cache_hits.inc()
        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")
    
    def record_cache_miss(self):
        try:
            cache_misses.inc()
        except Exception as e:
            logger.error(f"Error recording cache miss: {e}")
    
    def record_pcap_processing(self, status: str = "success"):
        try:
            pcap_files_processed.labels(status=status).inc()
        except Exception as e:
            logger.error(f"Error recording PCAP processing: {e}")
    
    def record_anomaly_detection(self, score: float = None):
        try:
            anomaly_detections.inc()
            if score is not None:
                network_anomaly_score.set(score)
        except Exception as e:
            logger.error(f"Error recording anomaly detection: {e}")
    
    def update_model_metrics(self, model_type: str, loaded: bool, accuracy: float = None, loss: float = None):
        try:
            model_loaded.labels(model_type=model_type).set(1 if loaded else 0)
            if accuracy is not None:
                model_accuracy.labels(model_type=model_type).set(accuracy)
            if loss is not None:
                model_loss.labels(model_type=model_type).set(loss)
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
    
    def update_network_metrics(self, packet_rate: float = None, byte_rate: float = None,
                              flow_counts: Dict[str, int] = None, 
                              traffic_volumes: Dict[str, float] = None):
        try:
            if packet_rate is not None:
                network_packet_rate.set(packet_rate)
            if byte_rate is not None:
                network_byte_rate.set(byte_rate)
            if flow_counts:
                for protocol, count in flow_counts.items():
                    network_flow_count.labels(protocol=protocol).set(count)
            if traffic_volumes:
                for direction, volume in traffic_volumes.items():
                    network_traffic_volume_bytes.labels(direction=direction).set(volume)
        except Exception as e:
            logger.error(f"Error updating network metrics: {e}")
    
    def get_metrics(self) -> bytes:
        try:
            self._update_process_metrics()
            self._update_system_metrics()
            
            if PROMETHEUS_MULTIPROC_DIR and os.path.exists(PROMETHEUS_MULTIPROC_DIR):
                from prometheus_client import multiprocess, CollectorRegistry
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                return generate_latest(registry)
            else:
                return generate_latest(REGISTRY)
                
        except Exception as e:
            logger.error(f"Error generating metrics: {e}", exc_info=True)
            return b"# Metrics generation error\n# TYPE api_up gauge\napi_up 0\n"
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        
        try:
            self._update_process_metrics()
            self._update_system_metrics()
            
            cpu_times = self.process.cpu_times()
            memory_info = self.process.memory_info()
            
            metrics = {
                'uptime_seconds': uptime,
                'multiprocess': self.multiprocess,
                'process': {
                    'pid': self.process.pid,
                    'cpu_seconds_total': cpu_times.user + cpu_times.system,
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_rss_mb': memory_info.rss / (1024 * 1024),
                    'memory_vms_mb': memory_info.vms / (1024 * 1024),
                    'num_threads': self.process.num_threads(),
                    'num_connections': len(self.process.connections()) if hasattr(self.process, 'connections') else 0,
                },
                'system': {
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'gpu_available': torch.cuda.is_available()
                }
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting metrics dict: {e}", exc_info=True)
            return {
                'error': str(e),
                'uptime_seconds': uptime
            }

metrics_collector = MetricsCollector()

try:
    api_process_cpu_percent.set(0)
    api_process_memory_mb.set(0)
    system_cpu_usage_percent.set(0)
    cache_hits.inc(0)
    cache_misses.inc(0)
    logger.info("Metrics initialized with default values")
except Exception as e:
    logger.error(f"Error initializing metrics: {e}")