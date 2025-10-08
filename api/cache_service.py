import json
import hashlib
import asyncio
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class CacheService:

    def __init__(self):
        self.cache = {}
        self.max_size = 1000
    
    async def connect(self):
        logger.info("Cache service initialized")
        return True
    
    async def disconnect(self):
        self.cache.clear()
        logger.info("Cache service disconnected")
    
    async def ping(self) -> bool:
        return True
    
    def generate_key(self, data: dict) -> str:
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
        