
'''
Run api/generatetraffic.py before testing moitoring performance
'''

import requests
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class MonitoringPerformanceAnalyzer:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.api_url = "http://localhost:8000"
        self.grafana_url = "http://localhost:3000"
        
    def query_prometheus(self, query, start=None, end=None, step="15s"):
        """Execute Prometheus query"""
        try:
            if start and end:
                # Range query
                params = {
                    'query': query,
                    'start': start.isoformat() + 'Z',
                    'end': end.isoformat() + 'Z',
                    'step': step
                }
                response = requests.get(f"{self.prometheus_url}/api/v1/query_range", params=params)
            else:
                # Instant query
                params = {'query': query}
                response = requests.get(f"{self.prometheus_url}/api/v1/query", params=params)
            
            return response.json()
        except Exception as e:
            print(f"Query failed: {e}")
            return {'status': 'error', 'data': {'result': []}}
    
    def analyze_metrics_collection_overhead(self):
        """Analyze the overhead of metrics collection"""
        print("\n" + "="*60)
        print("METRICS COLLECTION PERFORMANCE ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Scrape Performance - check for the actual job names
        scrape_queries = [
            ('up{job="network-traffic-api"}', 'API'),
            ('up{job="prometheus"}', 'Prometheus'),
            ('up{job="node"}', 'Node Exporter'),
            ('scrape_duration_seconds{job="network-traffic-api"}', 'API Scrape Duration'),
            ('scrape_duration_seconds{job="node"}', 'Node Scrape Duration')
        ]
        
        for query, name in scrape_queries:
            data = self.query_prometheus(query)
            if data['status'] == 'success' and data['data']['result']:
                value = float(data['data']['result'][0]['value'][1])
                if 'duration' in name.lower():
                    results[f'{name.replace(" ", "_").lower()}_ms'] = value * 1000
                    print(f"{name}: {value * 1000:.2f}ms")
                else:
                    results[f'{name.replace(" ", "_").lower()}_up'] = value
                    print(f"{name} Up: {'Yes' if value == 1 else 'No'}")
        
        # Number of Active Series
        series_query = 'prometheus_tsdb_symbol_table_size_bytes'
        series_data = self.query_prometheus(series_query)
        
        if series_data['status'] == 'success' and series_data['data']['result']:
            active_series = int(float(series_data['data']['result'][0]['value'][1]))
            results['active_series_bytes'] = active_series
            print(f"Symbol Table Size: {active_series/1024:.2f} KB")
        
        # Ingestion Rate
        ingestion_query = 'rate(prometheus_tsdb_head_samples_appended_total[5m])'
        ingestion_data = self.query_prometheus(ingestion_query)
        
        if ingestion_data['status'] == 'success' and ingestion_data['data']['result']:
            ingestion_rate = float(ingestion_data['data']['result'][0]['value'][1])
            results['ingestion_rate'] = ingestion_rate
            print(f"Sample Ingestion Rate: {ingestion_rate:.2f} samples/sec")
        
        # Storage Size
        storage_query = 'prometheus_tsdb_storage_blocks_bytes'
        storage_data = self.query_prometheus(storage_query)
        
        if storage_data['status'] == 'success' and storage_data['data']['result']:
            storage_bytes = float(storage_data['data']['result'][0]['value'][1])
            results['storage_mb'] = storage_bytes / (1024 * 1024)
            print(f"Storage Size: {results['storage_mb']:.2f} MB")
        
        # Memory Usage for Prometheus
        memory_query = 'process_resident_memory_bytes{job="prometheus"}'
        memory_data = self.query_prometheus(memory_query)
        
        if memory_data['status'] == 'success' and memory_data['data']['result']:
            memory_bytes = float(memory_data['data']['result'][0]['value'][1])
            results['memory_mb'] = memory_bytes / (1024 * 1024)
            print(f"Prometheus Memory Usage: {results['memory_mb']:.2f} MB")
        
        # API Metrics Collection
        api_metrics_queries = [
            ('api_requests_total', 'Total API Requests'),
            ('predictions_total', 'Total Predictions'),
            ('rate(api_request_duration_seconds_count[5m])', 'API Request Rate')
        ]
        
        for query, name in api_metrics_queries:
            data = self.query_prometheus(query)
            if data['status'] == 'success' and data['data']['result']:
                if isinstance(data['data']['result'], list) and len(data['data']['result']) > 0:
                    value = float(data['data']['result'][0]['value'][1])
                    results[name.replace(" ", "_").lower()] = value
                    print(f"{name}: {value:.2f}")
        
        return results
    
    def analyze_dashboard_performance(self):
        """Analyze Grafana dashboard rendering performance"""
        print("\n" + "="*60)
        print("DASHBOARD RENDERING PERFORMANCE")
        print("="*60)
        
        results = {}
        
        # Test dashboard query performance
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        test_queries = [
            ('api_requests_total', 'API Request Count'),
            ('rate(api_request_duration_seconds_bucket[5m])', 'Request Duration'),
            ('predictions_total', 'Predictions Count'),
            ('system_cpu_usage_percent', 'CPU Usage'),
            ('network_packet_rate', 'Packet Rate'),
            ('network_anomaly_score', 'Anomaly Score')
        ]
        
        for query, name in test_queries:
            start = time.time()
            response = self.query_prometheus(query, start_time, end_time)
            query_time = (time.time() - start) * 1000
            
            if response['status'] == 'success':
                data_points = sum(len(r.get('values', [])) for r in response['data']['result'])
                results[f'{name.replace(" ", "_").lower()}_query_ms'] = query_time
                results[f'{name.replace(" ", "_").lower()}_data_points'] = data_points
                print(f"{name}: {query_time:.2f}ms for {data_points} data points")
        
        return results
    
    def analyze_monitoring_stack_resources(self):
        """Analyze resource consumption by monitoring stack"""
        print("\n" + "="*60)
        print("MONITORING STACK RESOURCE CONSUMPTION")
        print("="*60)
        
        results = {}
        
        # Check actual running containers using the correct job names
        components = [
            ('prometheus', 'prometheus'),
            ('network-traffic-api', 'api'),
            ('node', 'node-exporter')
        ]
        
        for job_name, display_name in components:
            # Check if component is up
            up_query = f'up{{job="{job_name}"}}'
            up_data = self.query_prometheus(up_query)
            
            if up_data['status'] == 'success' and up_data['data']['result']:
                is_up = float(up_data['data']['result'][0]['value'][1])
                results[f'{display_name}_up'] = is_up
                
                if is_up == 1:
                    # For API use the custom metrics with the correct job label
                    if display_name == 'api':
                        # Query with the correct job label
                        cpu_query = f'api_process_cpu_percent{{job="{job_name}"}}'
                        cpu_data = self.query_prometheus(cpu_query)
                        
                        if cpu_data['status'] == 'success' and cpu_data['data']['result']:
                            cpu_percent = float(cpu_data['data']['result'][0]['value'][1])
                            results[f'{display_name}_cpu_percent'] = cpu_percent
                            print(f"{display_name.capitalize()} CPU: {cpu_percent:.2f}%")
                        else:
                            # Fallback: Try without job label (in case it's exported differently)
                            cpu_query = 'api_process_cpu_percent'
                            cpu_data = self.query_prometheus(cpu_query)
                            if cpu_data['status'] == 'success' and cpu_data['data']['result']:
                                cpu_percent = float(cpu_data['data']['result'][0]['value'][1])
                                results[f'{display_name}_cpu_percent'] = cpu_percent
                                print(f"{display_name.capitalize()} CPU: {cpu_percent:.2f}%")
                            else:
                                # Try standard process metrics as last resort
                                cpu_query = f'rate(process_cpu_seconds_total{{job="{job_name}"}}[5m]) * 100'
                                cpu_data = self.query_prometheus(cpu_query)
                                if cpu_data['status'] == 'success' and cpu_data['data']['result']:
                                    cpu_percent = float(cpu_data['data']['result'][0]['value'][1])
                                    results[f'{display_name}_cpu_percent'] = cpu_percent
                                    print(f"{display_name.capitalize()} CPU: {cpu_percent:.2f}%")
                                else:
                                    print(f"{display_name.capitalize()} CPU: Metrics not available")
                                    results[f'{display_name}_cpu_percent'] = 0
                        
                        # Memory metrics with correct job label
                        mem_query = f'api_process_memory_mb{{job="{job_name}"}}'
                        mem_data = self.query_prometheus(mem_query)
                        
                        if mem_data['status'] == 'success' and mem_data['data']['result']:
                            memory_mb = float(mem_data['data']['result'][0]['value'][1])
                            results[f'{display_name}_memory_mb'] = memory_mb
                            print(f"{display_name.capitalize()} Memory: {memory_mb:.2f} MB")
                        else:
                            #Try without job label
                            mem_query = 'api_process_memory_mb'
                            mem_data = self.query_prometheus(mem_query)
                            if mem_data['status'] == 'success' and mem_data['data']['result']:
                                memory_mb = float(mem_data['data']['result'][0]['value'][1])
                                results[f'{display_name}_memory_mb'] = memory_mb
                                print(f"{display_name.capitalize()} Memory: {memory_mb:.2f} MB")
                            else:
                                # Last resort: standard process metrics
                                mem_query = f'process_resident_memory_bytes{{job="{job_name}"}}'
                                mem_data = self.query_prometheus(mem_query)
                                if mem_data['status'] == 'success' and mem_data['data']['result']:
                                    memory_bytes = float(mem_data['data']['result'][0]['value'][1])
                                    memory_mb = memory_bytes / (1024 * 1024)
                                    results[f'{display_name}_memory_mb'] = memory_mb
                                    print(f"{display_name.capitalize()} Memory: {memory_mb:.2f} MB")
                                else:
                                    print(f"{display_name.capitalize()} Memory: Metrics not available")
                                    results[f'{display_name}_memory_mb'] = 0
                    else:
                        # For other components, use standard process metrics
                        cpu_queries = [
                            f'rate(process_cpu_seconds_total{{job="{job_name}"}}[5m]) * 100',
                            f'rate(container_cpu_usage_seconds_total{{job="{job_name}"}}[5m]) * 100',
                        ]
                        
                        cpu_percent = 0
                        for cpu_query in cpu_queries:
                            cpu_data = self.query_prometheus(cpu_query)
                            if cpu_data['status'] == 'success' and cpu_data['data']['result']:
                                cpu_percent = float(cpu_data['data']['result'][0]['value'][1])
                                break
                        
                        if cpu_percent > 0:
                            results[f'{display_name}_cpu_percent'] = cpu_percent
                            print(f"{display_name.capitalize()} CPU: {cpu_percent:.2f}%")
                        else:
                            print(f"{display_name.capitalize()} CPU: Metrics not available")
                        
                        # Memory metrics
                        mem_query = f'process_resident_memory_bytes{{job="{job_name}"}}'
                        mem_data = self.query_prometheus(mem_query)
                        
                        if mem_data['status'] == 'success' and mem_data['data']['result']:
                            memory_bytes = float(mem_data['data']['result'][0]['value'][1])
                            memory_mb = memory_bytes / (1024 * 1024)
                            results[f'{display_name}_memory_mb'] = memory_mb
                            print(f"{display_name.capitalize()} Memory: {memory_mb:.2f} MB")
                        else:
                            print(f"{display_name.capitalize()} Memory: Metrics not available")
            else:
                print(f"{display_name.capitalize()}: Not running or not scraped")
        
        # Check Grafana
        try:
            response = requests.get(f"{self.grafana_url}/api/health", timeout=2)
            if response.status_code == 200:
                results['grafana_up'] = 1
                print("Grafana: Running (metrics not exposed)")
            else:
                results['grafana_up'] = 0
                print("Grafana: Not healthy")
        except:
            results['grafana_up'] = 0
            print("Grafana: Not accessible")
        
        # Total monitoring overhead
        total_cpu = sum(v for k, v in results.items() if 'cpu_percent' in k)
        total_memory = sum(v for k, v in results.items() if 'memory_mb' in k)
        
        results['total_monitoring_cpu_percent'] = total_cpu
        results['total_monitoring_memory_mb'] = total_memory
        
        print(f"\nTotal Monitoring Stack:")
        print(f"  CPU: {total_cpu:.2f}%")
        print(f"  Memory: {total_memory:.2f} MB")
        
        return results

    def run_load_test_with_metrics(self, duration_seconds=60):
        """Run load test while monitoring metrics collection overhead"""
        print("\n" + "="*60)
        print("LOAD TEST WITH METRICS MONITORING")
        print("="*60)
        
        import asyncio
        import httpx
        
        async def send_requests():
            async with httpx.AsyncClient() as client:
                tasks = []
                for _ in range(100):
                    task = client.post(
                        f"{self.api_url}/predict",
                        json={
                            "sequence_data": [[0.5] * 8] * 10,
                            "time_window": "30s",
                            "prediction_horizon": 1
                        },
                        timeout=10.0
                    )
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                return successful
        
        #  Baseline metrics before load
        print("Collecting baseline metrics...")
        baseline_metrics = self.analyze_metrics_collection_overhead()
        
        # Run load test
        print(f"\nRunning load test for {duration_seconds} seconds...")
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            successful = asyncio.run(send_requests())
            request_count += successful
            print(f"  Sent {request_count} requests...", end='\r')
        
        print(f"\nTotal requests sent: {request_count}")
        
        # Metrics after load
        print("\nCollecting post-load metrics...")
        time.sleep(10)  # Wait for metrics to update
        post_load_metrics = self.analyze_metrics_collection_overhead()
        
        # Calculate overhead
        overhead = {}
        for key in baseline_metrics:
            if key in post_load_metrics:
                overhead[f'{key}_increase'] = post_load_metrics[key] - baseline_metrics[key]
                if baseline_metrics[key] > 0:
                    overhead[f'{key}_percent_increase'] = ((post_load_metrics[key] - baseline_metrics[key]) / baseline_metrics[key]) * 100
        
        print("\n" + "="*40)
        print("METRICS COLLECTION OVERHEAD UNDER LOAD")
        print("="*40)
        
        for key, value in overhead.items():
            if 'percent_increase' in key:
                metric_name = key.replace('_percent_increase', '')
                print(f"{metric_name}: {value:.2f}% increase")
        
        return overhead
    
    def generate_report(self, output_dir="api/monitoring"):
        """Generate comprehensive monitoring performance report"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE MONITORING REPORT")
        print("="*60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics_collection': self.analyze_metrics_collection_overhead(),
            'dashboard_performance': self.analyze_dashboard_performance(),
            'stack_resources': self.analyze_monitoring_stack_resources()
        }
        
        #Save report
        output_file = output_path / "monitoring_performance_report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nReport saved to {output_file}")
        
        #Create visualization
        self.visualize_results(report, output_path)
        
        return report
    
    def visualize_results(self, report, output_path):
        """Create visualization of monitoring performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        #Resource Consumption
        ax = axes[0, 0]
        components = ['prometheus', 'api', 'node-exporter']
        cpu_values = []
        memory_values = []
        
        for c in components:
            cpu_key = f'{c}_cpu_percent'
            mem_key = f'{c}_memory_mb'
            cpu_values.append(report['stack_resources'].get(cpu_key, 0))
            memory_values.append(report['stack_resources'].get(mem_key, 0))
        
        x = range(len(components))
        width = 0.35
        ax.bar([i - width/2 for i in x], cpu_values, width, label='CPU %', color='blue')
        ax.bar([i + width/2 for i in x], [m/100 for m in memory_values], width, label='Memory (100MB)', color='orange')
        ax.set_xlabel('Component')
        ax.set_ylabel('Usage')
        ax.set_title('Monitoring Stack Resource Consumption')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        #Query Performance
        ax = axes[0, 1]
        query_times = {k.replace('_query_ms', ''): v 
                      for k, v in report['dashboard_performance'].items() 
                      if '_query_ms' in k}
        if query_times:
            bars = ax.bar(range(len(query_times)), list(query_times.values()), color='green')
            ax.set_xlabel('Query')
            ax.set_ylabel('Response Time (ms)')
            ax.set_title('Dashboard Query Performance')
            ax.set_xticks(range(len(query_times)))
            ax.set_xticklabels(list(query_times.keys()), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Adding value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        #Metrics Collection Stats
        ax = axes[1, 0]
        metrics_stats = report['metrics_collection']
        stats_to_plot = {}
        
        if 'api_scrape_duration_ms' in metrics_stats:
            stats_to_plot['API Scrape'] = metrics_stats['api_scrape_duration_ms']
        if 'node_scrape_duration_ms' in metrics_stats:
            stats_to_plot['Node Scrape'] = metrics_stats['node_scrape_duration_ms']
        if 'ingestion_rate' in metrics_stats:
            stats_to_plot['Ingestion Rate\n(samples/s)'] = metrics_stats['ingestion_rate']
        
        if stats_to_plot:
            bars = ax.bar(stats_to_plot.keys(), stats_to_plot.values(), color='purple')
            ax.set_ylabel('Value')
            ax.set_title('Metrics Collection Statistics')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # Summary Text
        ax = axes[1, 1]
        ax.axis('off')
        
        # Component status
        components_up = []
        for comp in ['api', 'prometheus', 'node-exporter', 'grafana']:
            if report['stack_resources'].get(f'{comp}_up', 0) == 1:
                components_up.append(comp)
        
        summary_text = f"""MONITORING PERFORMANCE SUMMARY
            ===============================
            Active Components: {', '.join(components_up)}

            Resource Usage:
            Total CPU: {report['stack_resources'].get('total_monitoring_cpu_percent', 0):.2f}%
            Total Memory: {report['stack_resources'].get('total_monitoring_memory_mb', 0):.2f} MB

            Metrics Collection:
            Ingestion Rate: {metrics_stats.get('ingestion_rate', 0):.2f} samples/sec
            Storage Size: {metrics_stats.get('storage_mb', 0):.2f} MB
            Prometheus Memory: {metrics_stats.get('memory_mb', 0):.2f} MB

            API Metrics:
            Total Requests: {metrics_stats.get('total_api_requests', 0):.0f}
            Total Predictions: {metrics_stats.get('total_predictions', 0):.0f}

            Performance Assessment:
            Stack Health: {'Good' if len(components_up) >= 3 else 'Degraded'}
            Resource Usage: {'Acceptable' if report['stack_resources'].get('total_monitoring_cpu_percent', 0) < 20 else 'High'}
            Query Performance: {'Good' if all(v < 500 for k, v in report['dashboard_performance'].items() if '_query_ms' in k) else 'Needs optimization'}
            """
        ax.text(0.05, 0.5, summary_text, fontsize=9, family='monospace', va='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_path / 'monitoring_performance.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {fig_path}")
        
        # Don't show in non-interactive environments
        try:
            plt.show()
        except:
            pass

def main():
    analyzer = MonitoringPerformanceAnalyzer()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Run load test
    analyzer.run_load_test_with_metrics(duration_seconds=30)
    
    print("\nKey Findings:")
    print(f"  - Components Running: {sum(1 for k, v in report['stack_resources'].items() if k.endswith('_up') and v == 1)}")
    print(f"  - Total CPU Usage: {report['stack_resources'].get('total_monitoring_cpu_percent', 0):.2f}%")
    print(f"  - Total Memory: {report['stack_resources'].get('total_monitoring_memory_mb', 0):.2f} MB")
    print(f"  - Metrics Ingestion: {report['metrics_collection'].get('ingestion_rate', 0):.2f} samples/sec")
    
    # Check if all components are running
    missing_components = []
    for comp in ['api', 'prometheus', 'node-exporter']:
        if report['stack_resources'].get(f'{comp}_up', 0) != 1:
            missing_components.append(comp)
    
    if missing_components:
        print(f"\n WARNING: The following components are not running: {', '.join(missing_components)}")

if __name__ == "__main__":
    main()