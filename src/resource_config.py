from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

class ResourceConfig(BaseModel):
    observation_period: float = Field(os.getenv("OBSERVATION_PERIOD", 10.0), description="Duration in seconds to observe before pruning.")
    check_interval: float = Field(os.getenv("CHECK_INTERVAL", 1.0), description="Interval in seconds between resource usage checks.")
    rolling_window_size: int = Field(os.getenv("ROLLING_WINDOW_SIZE", 20), description="Number of samples to keep for rolling averages.")
    target_utilization: float = Field(os.getenv("TARGET_UTILIZATION", 0.9), description="Target fraction (~0.9 = 90%) of resource utilization.")
    enabled: bool = Field(os.getenv("ENABLED", True), description="Enable or disable the dynamic resource manager.")
    max_memory_usage_mb: float = Field(os.getenv("MAX_MEMORY_USAGE_MB", 8192.0), description="Maximum memory usage in MB.")
    max_cpu_usage_percent: float = Field(os.getenv("MAX_CPU_USAGE_PERCENT", 90.0), description="Maximum CPU usage in percent.")
    max_gpu_usage_percent: float = Field(os.getenv("MAX_GPU_USAGE_PERCENT", 90.0), description="Maximum GPU usage in percent.")

