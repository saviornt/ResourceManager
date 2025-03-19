import psutil
import time
import logging
from .resource_config import ResourceConfig
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlShutdown
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

resource_logger = logging.getLogger("resource_manager")

# NVML Context Manager
class NVMLContext:
    """
    A simple context manager for NVML.
    Call __enter__() to initialize NVML and __exit__() to shutdown.
    """
    def __init__(self):
        self.initialized = False
    
    def __enter__(self):
        try:
            nvmlInit()
            self.initialized = True
            resource_logger.info("NVML initialized successfully")
        except Exception as e:
            resource_logger.error(f"Failed to initialize NVML: {e}")
            raise
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.initialized:
            try:
                nvmlShutdown()
                resource_logger.info("NVML shutdown successfully")
            except Exception as e:
                resource_logger.error(f"Failed to shutdown NVML: {e}")
                
class ResourceManager:
    def __init__(self,
                 config: ResourceConfig = None,
                 max_memory_usage_mb: float = None,
                 max_cpu_usage_percent: float = None,
                 max_gpu_usage_percent: float = None,
                 check_interval: float = None):
        self.config = config or ResourceConfig()
        self.check_interval = check_interval if check_interval is not None else self.config.check_interval

        self.max_memory_usage_mb = max_memory_usage_mb if max_memory_usage_mb is not None else 1024.0
        self.max_cpu_usage_percent = max_cpu_usage_percent if max_cpu_usage_percent is not None else 90.0
        self.max_gpu_usage_percent = max_gpu_usage_percent if max_gpu_usage_percent is not None else 90.0
        
        self.last_check = time.time()
        
        # CPU caching attributes
        self._cached_cpu_percent = None
        self._cpu_last_measure_time = 0

        # NVML Initialization and GPU handle
        self._nvml_context = None
        self.gpu_handle = None
        if NVML_AVAILABLE:
            try:
                self._nvml_context = NVMLContext()
                self._nvml_context.__enter__()
                self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
                resource_logger.info("GPU handle aquired successfully")
            except Exception as e:
                resource_logger.error(f"Failed to initialize NVML: {e}")
                self.gpu_handle = None
            
    def _get_cpu_percent(self) -> float:
        """
        Get the current CPU usage percentage.
        Caches the result to avoid frequent calls to psutil.
        """
        current_time = time.time()
        if (current_time - self._cpu_last_measure_time) < self.check_interval and self._cached_cpu_percent is not None:
            return self._cached_cpu_percent
        cpu_usage = psutil.cpu_percent(interval=0.05)
        self._cached_cpu_percent = cpu_usage
        self._cpu_last_measure_time = current_time
        return cpu_usage      

    def should_prune(self) -> bool:
        """
        Checks if current resource usage exceeds limits.
        Returns True if any resource (memory, CPU, or GPU) usage exceeds the set threshold.
        """
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False
        self.last_check = current_time

        process = psutil.Process()
        try:
            mem_info = process.memory_info()
            mem_usage_mb = mem_info.rss / (1024 * 1024)
        except psutil.Error as e:
            resource_logger.error("Error getting process memory info: %s", e)
            mem_usage_mb = 0

        cpu_usage = self._get_cpu_percent()

        gpu_usage = 0
        if self.gpu_handle:
            try:
                util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_usage = util.gpu
            except Exception as e:
                resource_logger.error("Error getting GPU utilization: %s", e)
                gpu_usage = 0

        resource_logger.info(f"CPU={cpu_usage:.2f}% | MEM={mem_usage_mb:.2f}MB | GPU={gpu_usage:.2f}%")
        if (mem_usage_mb > self.max_memory_usage_mb or
            cpu_usage > self.max_cpu_usage_percent or
            gpu_usage > self.max_gpu_usage_percent):
            return True
        return False

    def close(self):
        """
        Clean up NVML context if it was initialized.
        """
        if NVML_AVAILABLE and self._nvml_context:
            try:
                self._nvml_context.__exit__(None, None, None)
            except Exception as e:
                resource_logger.error("Error during NVML shutdown: %s", e)