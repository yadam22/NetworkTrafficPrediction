# Module to generate traffic across the API target 
import asyncio
import httpx
import random
import time
import json
from datetime import datetime
import numpy as np

class MonitoringTrafficGenerator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'total_duration': 0
        }
    
    async def generate_predict_traffic(self, count=100):        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(count):
                sequence_length = random.randint(5, 20)
                feature_count = 8
                
                request_data = {
                    "sequence_data": [
                        [random.gauss(0.5, 0.2) for _ in range(feature_count)]
                        for _ in range(sequence_length)
                    ],
                    "time_window": random.choice(["10s", "30s", "1min"]),
                    "prediction_horizon": random.randint(1, 5),
                    "task": random.choice(["traffic_prediction", "anomaly_detection", "flow_prediction"]),
                    "return_confidence": random.choice([True, False])
                }
        
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"{self.base_url}/predict",
                        json=request_data
                    )
                    duration = time.time() - start_time
                    
                    self.stats['total_requests'] += 1
                    self.stats['total_duration'] += duration
                    
                    if response.status_code == 200:
                        self.stats['successful'] += 1
                        print(f" Request {i+1}/{count} - {duration*1000:.2f}ms - {response.status_code}")
                    else:
                        self.stats['failed'] += 1
                        print(f" Request {i+1}/{count} - {response.status_code}")
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f" Request {i+1}/{count} failed: {str(e)[:50]}")
                
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
    
    async def generate_batch_traffic(self, count=20):      
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(count):
                batch_size = random.randint(2, 10)
                
                sequences = []
                for _ in range(batch_size):
                    sequences.append({
                        "sequence_data": [
                            [random.random() for _ in range(8)]
                            for _ in range(random.randint(5, 15))
                        ],
                        "time_window": random.choice(["10s", "30s", "1min"]),
                        "prediction_horizon": random.randint(1, 3)
                    })
                
                request_data = {
                    "sequences": sequences,
                    "parallel_processing": random.choice([True, False]),
                    "max_batch_size": 32
                }
                
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"{self.base_url}/predict/batch",
                        json=request_data
                    )
                    duration = time.time() - start_time
                    
                    self.stats['total_requests'] += 1
                    self.stats['total_duration'] += duration
                    
                    if response.status_code == 200:
                        self.stats['successful'] += 1
                        print(f" Batch {i+1}/{count} ({batch_size} items) - {duration*1000:.2f}ms")
                    else:
                        self.stats['failed'] += 1
                        print(f" Batch {i+1}/{count} - {response.status_code}")
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f" Batch {i+1}/{count} failed: {str(e)[:50]}")
                
                await asyncio.sleep(0.5)
    
    async def generate_anomaly_traffic(self, count=30):
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(count):
                num_samples = random.randint(10, 100)
                
                traffic_data = []
                for _ in range(num_samples):
                    # Generate some normal and some anomalous patterns
                    if random.random() < 0.1:  # 10% anomalies
                        # Anomalous traffic
                        traffic_data.append({
                            "packet_count": float(random.randint(50000, 100000)),
                            "total_bytes": float(random.randint(5000000, 10000000)),
                            "packet_rate": float(random.uniform(5000, 10000)),
                            "byte_rate": float(random.uniform(500000, 1000000))
                        })
                    else:
                        # Normal traffic
                        traffic_data.append({
                            "packet_count": float(random.randint(100, 5000)),
                            "total_bytes": float(random.randint(1000, 500000)),
                            "packet_rate": float(random.uniform(10, 500)),
                            "byte_rate": float(random.uniform(1000, 50000))
                        })
                
                request_data = {
                    "traffic_data": traffic_data,
                    "sensitivity": random.uniform(0.3, 0.8)
                }
                
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"{self.base_url}/anomaly/detect",
                        json=request_data
                    )
                    duration = time.time() - start_time
                    
                    self.stats['total_requests'] += 1
                    self.stats['total_duration'] += duration
                    
                    if response.status_code == 200:
                        result = response.json()
                        anomalies_found = result.get('anomaly_count', 0)
                        self.stats['successful'] += 1
                        print(f" Anomaly {i+1}/{count} - Found {anomalies_found} anomalies - {duration*1000:.2f}ms")
                    else:
                        self.stats['failed'] += 1
                        print(f" Anomaly {i+1}/{count} - {response.status_code}")
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f" Anomaly {i+1}/{count} failed: {str(e)[:50]}")
                
                await asyncio.sleep(0.2)
    
    async def generate_websocket_traffic(self, messages=50):
        import websockets
        
        try:
            uri = f"ws://localhost:8000/ws/predict"
            async with websockets.connect(uri) as websocket:
                for i in range(messages):
                    message = {
                        "sequence_data": [
                            [random.random() for _ in range(8)]
                            for _ in range(random.randint(5, 15))
                        ],
                        "time_window": "30s",
                        "prediction_horizon": 1
                    }
                    
                    try:
                        start_time = time.time()
                        await websocket.send(json.dumps(message))
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        duration = time.time() - start_time
                        
                        self.stats['total_requests'] += 1
                        self.stats['total_duration'] += duration
                        self.stats['successful'] += 1
                        
                        print(f"WS Message {i+1}/{messages} - {duration*1000:.2f}ms")
                        
                    except asyncio.TimeoutError:
                        self.stats['failed'] += 1
                        print(f" WS Message {i+1}/{messages} - Timeout")
                    
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f" WebSocket connection failed: {e}")
    
    async def generate_mixed_traffic(self, duration_minutes=5):     
        end_time = time.time() + (duration_minutes * 60)
        
        tasks = []
        while time.time() < end_time:
            # Mix different types of requests
            tasks = [
                self.generate_predict_traffic(20),
                self.generate_batch_traffic(5),
                self.generate_anomaly_traffic(10)
            ]
            
            await asyncio.gather(*tasks)
            
            remaining = (end_time - time.time()) / 60
            print(f"\n  {remaining:.1f} minutes remaining...")
            
            await asyncio.sleep(5)
        
        return self.stats
    
    async def health_check_loop(self, interval=10, count=30):      
        async with httpx.AsyncClient() as client:
            for i in range(count):
                try:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        health = response.json()
                        print(f" Health {i+1}/{count}: {health['status']}")
                except Exception as e:
                    print(f" Health check failed: {e}")
                
                await asyncio.sleep(interval)
    
    async def generate_burst_traffic(self, burst_size=100, bursts=5):   
        for burst_num in range(bursts):
            print(f"\n Burst {burst_num + 1}/{bursts} starting...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                tasks = []
                for _ in range(burst_size):
                    task = client.post(
                        f"{self.base_url}/predict",
                        json={
                            "sequence_data": [[random.random() for _ in range(8)] for _ in range(10)],
                            "time_window": "30s",
                            "prediction_horizon": 1
                        }
                    )
                    tasks.append(task)
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time
                
                successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                failed = burst_size - successful
                
                self.stats['total_requests'] += burst_size
                self.stats['successful'] += successful
                self.stats['failed'] += failed
                
                print(f" Burst complete: {successful}/{burst_size} successful in {duration:.2f}s")
                print(f"Rate: {burst_size/duration:.2f} req/sec")
                
                # Cool down between bursts
                await asyncio.sleep(5)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("TRAFFIC GENERATION SUMMARY")
        print("="*60)
        print(f"Total Requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['successful'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_requests']) * 100
            avg_duration = (self.stats['total_duration'] / self.stats['successful']) * 1000
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Response Time: {avg_duration:.2f}ms")
        
async def main():
    generator = MonitoringTrafficGenerator()

    await generator.health_check_loop(interval=2, count=5)
    
    await generator.generate_predict_traffic(50)
    await generator.generate_batch_traffic(10)
    await generator.generate_anomaly_traffic(20)
    
    try:
        await generator.generate_websocket_traffic(20)
    except:
        print("WebSocket not available")
    
    await generator.generate_burst_traffic(burst_size=50, bursts=3)
    
    await generator.generate_mixed_traffic(duration_minutes=2)
    
    generator.print_summary()

if __name__ == "__main__":
    asyncio.run(main())

