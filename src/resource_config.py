from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os
import logging

resource_config_logger = logging.getLogger("resource_config")

load_dotenv()

if not os.path.exists(".env"):
    resource_config_logger.warning(".env file not found. Using default values or environment variables.")

class ResourceConfig(BaseModel):
    """
    Configuration settings for ResourceMonitor and ResourceManager.
    All settings can be overridden via environment variables.
    """
    # General settings
    observation_period: float = Field(float(os.getenv("OBSERVATION_PERIOD", 10.0)), description="Duration in seconds to observe resource utilization history before making pruning decisions.")
    check_interval: float = Field(float(os.getenv("CHECK_INTERVAL", 1.0)), description="Interval in seconds between resource usage checks.")
    rolling_window_size: int = Field(int(os.getenv("ROLLING_WINDOW_SIZE", 20)), description="Number of samples to keep for rolling averages (not directly used in dynamic pruning).") # Rolling window is less relevant with observation period logic
    target_utilization: float = Field(float(os.getenv("TARGET_UTILIZATION", 0.9)), description="Target fraction (~0.9 = 90%) of resource utilization to trigger dynamic pruning consideration.")
    enabled: bool = Field(os.getenv("ENABLED", "True").lower() == "true", description="Enable or disable the dynamic resource manager.")
    
    # Resource thresholds
    max_memory_usage_mb: float = Field(float(os.getenv("MAX_MEMORY_USAGE_MB", 8192.0)), description="Maximum memory usage in MB. Pruning is considered if average utilization over observation period is above target AND current usage exceeds this max.")
    max_cpu_usage_percent: float = Field(float(os.getenv("MAX_CPU_USAGE_PERCENT", 90.0)), description="Maximum CPU usage in percent. Pruning is considered if average utilization over observation period is above target AND current usage exceeds this max.")
    max_gpu_usage_percent: float = Field(float(os.getenv("MAX_GPU_USAGE_PERCENT", 90.0)), description="Maximum GPU usage in percent. Pruning is considered if average utilization over observation period is above target AND current usage exceeds this max.")
    max_network_usage_bytes_per_second: float = Field(float(os.getenv("MAX_NETWORK_USAGE_BYTES_PER_SECOND", 10 * 1024 * 1024.0)), description="Maximum network usage in bytes per second. Pruning is considered if average utilization over observation period is above target AND current usage exceeds this max.") # Default 10MB/s
    
    # Performance settings
    use_async: bool = Field(os.getenv("USE_ASYNC", "True").lower() == "true", description="Use asynchronous operations for better performance")
    cache_ttl: float = Field(float(os.getenv("CACHE_TTL", 1.0)), description="Cache time-to-live in seconds")
    cpu_sample_count: int = Field(int(os.getenv("CPU_SAMPLE_COUNT", 5)), description="Number of CPU samples to take for better accuracy")
    cpu_sample_interval: float = Field(float(os.getenv("CPU_SAMPLE_INTERVAL", 0.01)), description="Interval between CPU samples in seconds")
    thread_pool_workers: int = Field(int(os.getenv("THREAD_POOL_WORKERS", 4)), description="Number of workers in thread pool for CPU-bound tasks")
    
    # API settings
    api_cache_enabled: bool = Field(os.getenv("API_CACHE_ENABLED", "True").lower() == "true", description="Enable caching for API responses")
    api_cache_ttl: float = Field(float(os.getenv("API_CACHE_TTL", 0.5)), description="API cache time-to-live in seconds")
    enable_background_refresh: bool = Field(os.getenv("ENABLE_BACKGROUND_REFRESH", "True").lower() == "true", description="Enable background refresh of caches")

    @field_validator('observation_period', 'check_interval', 'cache_ttl', 'cpu_sample_interval', 'api_cache_ttl')
    @classmethod
    def validate_positive_time(cls, v: float) -> float:
        """Validates time-related fields are positive."""
        if v <= 0:
            raise ValueError(f"Time value must be positive, got {v}")
        return v
    
    @field_validator('rolling_window_size', 'cpu_sample_count', 'thread_pool_workers')
    @classmethod
    def validate_positive_integer(cls, v: int) -> int:
        """Validates integer fields are positive."""
        if v <= 0:
            raise ValueError(f"Integer value must be positive, got {v}")
        return v
    
    @field_validator('target_utilization')
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        """Validates utilization is a fraction between 0 and 1."""
        if not (0 < v <= 1.0):
            raise ValueError(f"Utilization must be between 0 and 1, got {v}")
        return v
    
    @field_validator('max_cpu_usage_percent', 'max_gpu_usage_percent')
    @classmethod
    def validate_percent(cls, v: float) -> float:
        """Validates percentage values."""
        if not (0 < v <= 100.0):
            raise ValueError(f"Percentage must be between 0 and 100, got {v}")
        return v
    
    @field_validator('max_memory_usage_mb', 'max_network_usage_bytes_per_second')
    @classmethod
    def validate_positive_resource_value(cls, v: float) -> float:
        """Validates resource values are positive."""
        if v <= 0:
            raise ValueError(f"Resource value must be positive, got {v}")
        return v