import asyncio
import json
import websockets
import logging
from typing import Dict, Any, List
import ssl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSocketPredictTester:
    def __init__(self, uri: str = "wss://network-traffic-api-845421504867.europe-west1.run.app/ws/predict"):
        self.uri = uri
        self.ssl_context = ssl.create_default_context()
        
    async def test_single_prediction(self):
        """Test a single prediction request"""
        try:
            async with websockets.connect(self.uri, ssl=self.ssl_context) as websocket:
                logger.info(f"Connected to {self.uri}")
                
                # Create a test prediction request
                request = {
                    "sequence_data": [
                        [100.5, 200.3, 150.2, 300.1, 250.5, 175.8, 225.4, 195.6],
                        [105.2, 195.8, 155.1, 295.5, 245.2, 180.3, 220.1, 200.3]
                    ],
                    "time_window": "30s",
                    "prediction_horizon": 1,
                    "task": "traffic_prediction",
                    "return_confidence": True
                }
                
                # Send the request
                await websocket.send(json.dumps(request))
                logger.info(f"Sent request: {json.dumps(request, indent=2)}")
                
                # Receive the response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                # Pretty print the response
                logger.info(f"Received response:")
                logger.info(json.dumps(response_data, indent=2))
                
                return response_data
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error during test: {e}")
            raise
    
    async def test_multiple_predictions(self, num_requests: int = 3):
        """Test multiple prediction requests in sequence"""
        try:
            async with websockets.connect(self.uri, ssl=self.ssl_context) as websocket:
                logger.info(f"Connected to {self.uri} for multiple requests")
                
                for i in range(num_requests):
                    request = {
                        "sequence_data": [
                            [100 + i*10 + j for j in range(8)],
                            [110 + i*10 + j for j in range(8)]
                        ],
                        "time_window": ["10s", "30s", "1min"][i % 3],
                        "prediction_horizon": i + 1,
                        "task": "traffic_prediction",
                        "return_confidence": True
                    }
                    
                    await websocket.send(json.dumps(request))
                    logger.info(f"\n--- Request {i+1}/{num_requests} ---")
                    logger.info(f"Sent: time_window={request['time_window']}, horizon={request['prediction_horizon']}")
                    
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if 'error' in response_data:
                        logger.error(f"Error in response: {response_data['error']}")
                    else:
                        logger.info(f"Predictions shape: {len(response_data.get('predictions', []))} samples")
                        logger.info(f"Confidence: {response_data.get('confidence', 'N/A')}")
                        logger.info(f"Timestamp: {response_data.get('timestamp', 'N/A')}")
                    
                    await asyncio.sleep(0.5)
                
                logger.info("\nAll requests completed successfully")
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error during multiple tests: {e}")
            raise
    
    async def test_error_handling(self):
        """Test error handling with invalid requests"""
        try:
            async with websockets.connect(self.uri, ssl=self.ssl_context) as websocket:
                logger.info(f"Testing error handling...")
                
                # Test invalid time window
                invalid_request = {
                    "sequence_data": [[1, 2, 3]],
                    "time_window": "invalid",  
                    "task": "traffic_prediction"
                }
                
                await websocket.send(json.dumps(invalid_request))
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Invalid time_window response: {response_data.get('error', 'No error')}")
                
                # Test invalid task
                invalid_request = {
                    "sequence_data": [[1, 2, 3]],
                    "task": "invalid_task" 
                }
                
                await websocket.send(json.dumps(invalid_request))
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Invalid task response: {response_data.get('error', 'No error')}")
                
                empty_request = {
                    "time_window": "30s"
                }
                
                await websocket.send(json.dumps(empty_request))
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Empty data response: {response_data.get('error', response_data.get('predictions', 'Got predictions'))}")
                
                logger.info("\nError handling tests completed")
                
        except Exception as e:
            logger.error(f"Error during error handling test: {e}")
            raise
    
    async def test_keep_alive(self, duration_seconds: int = 10):
        """Testing ws keep alive mechanism"""
        try:
            async with websockets.connect(self.uri, ssl=self.ssl_context) as websocket:
                logger.info(f"Testing keeping alive for {duration_seconds} seconds...")
                
                request = {
                    "sequence_data": [[1, 2, 3, 4, 5, 6, 7, 8]],
                    "time_window": "30s",
                    "task": "traffic_prediction"
                }
                
                await websocket.send(json.dumps(request))
                await websocket.recv()
                logger.info("Initial request successful")
                
                start_time = asyncio.get_event_loop().time()
                ping_count = 0
                
                while asyncio.get_event_loop().time() - start_time < duration_seconds:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=65)
                        data = json.loads(message)
                        
                        if 'ping' in data:
                            ping_count += 1
                            logger.info(f"Received keep alive ping #{ping_count}")
                        else:
                            logger.info(f"Received unexpected message: {data}")
                            
                    except asyncio.TimeoutError:
                        logger.info("No ping received in 65 seconds (expected)")
                        break
                
                logger.info(f"\nKeep alive test completed. Received {ping_count} pings")
                
        except Exception as e:
            logger.error(f"Error during keep alive test: {e}")
            raise
    
    async def test_connection(self):
        """Connection test"""
        try:
            logger.info(f"Testing connection to {self.uri}")
            async with websockets.connect(
                self.uri, 
                ssl=self.ssl_context,
                ping_interval=10,
                ping_timeout=5
            ) as websocket:
                logger.info("Successfully connected to WebSocket endpoint")
                
                # Try a simple request
                simple_request = {
                    "sequence_data": [[1, 2, 3, 4, 5, 6, 7, 8]],
                    "time_window": "30s",
                    "task": "traffic_prediction"
                }
                
                await websocket.send(json.dumps(simple_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                response_data = json.loads(response)
                
                if 'error' not in response_data:
                    logger.info("Successfully received prediction response")
                else:
                    logger.warning(f"Received error response: {response_data['error']}")
                    
                return True
                
        except asyncio.TimeoutError:
            logger.error("Connection timed out ")
            return False
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

async def main():

    tester = WebSocketPredictTester()
    
    print("=" * 60)
    print("WebSocket /ws/predict Endpoint Test Suite")
    print(f"Target: {tester.uri}")
    print("=" * 60)
    
    print("\n0. Testing connection...")
    connection_ok = await tester.test_connection()
    
    if not connection_ok:
        print("\nConnection test failed. Please check:")
        return
    
    await asyncio.sleep(1)
    
    print("\n1. Testing single prediction...")
    await tester.test_single_prediction()
    await asyncio.sleep(1)
    
    print("\n2. Testing multiple predictions...")
    await tester.test_multiple_predictions(num_requests=3)
    await asyncio.sleep(1)
    
    print("\n3. Testing error handling...")
    await tester.test_error_handling()
    await asyncio.sleep(1)
    
    print("\n4. Testing keep alive mechanism ...")
    await tester.test_keep_alive(duration_seconds=10) 
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    print("Starting WebSocket tests...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTests failed with error: {e}")