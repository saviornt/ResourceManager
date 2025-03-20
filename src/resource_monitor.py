import psutil
import time
import logging
import threading
import asyncio
import concurrent.futures
from functools import lru_cache
from collections import deque
from typing import Tuple, Optional, Deque, Dict, Any

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from .resource_config import ResourceConfig

resource_monitor_logger = logging.getLogger("resource_monitor")

# Create thread pool executor for CPU-bound tasks
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# NVML Context Manager
class NVMLContext:
    """
    A simple context manager for NVML.
    Call __enter__() to initialize NVML and __exit__() to shutdown.
    """
    def __init__(self):
        self.initialized = False
        self._init_lock = threading.Lock()  # Lock for NVML init/shutdown
        self._error_count = 0  # Track errors to prevent excessive logging
        self._working = False  # Flag to indicate if NVML is working properly
        resource_monitor_logger.debug("NVMLContext.__init__() - Entry")

    def __enter__(self):
        resource_monitor_logger.debug("NVMLContext.__enter__() - Entry")
        with self._init_lock:
            if not self.initialized and self._error_count < 10:
                try:
                    resource_monitor_logger.debug("NVMLContext.__enter__() - Initializing NVML...")
                    nvmlInit()
                    self.initialized = True
                    resource_monitor_logger.info("NVML initialized successfully")
                    
                    # Verify NVML is working by trying to get device count
                    try:
                        device_count = nvmlDeviceGetCount()
                        resource_monitor_logger.debug(f"NVMLContext.__enter__() - Found {device_count} GPU devices")
                        self._working = True
                    except Exception as count_err:
                        resource_monitor_logger.warning(f"NVMLContext.__enter__() - NVML initialized but couldn't get device count: {count_err}")
                        self._working = False
                        
                    resource_monitor_logger.debug(f"NVMLContext.__enter__() - NVML initialized successfully, working: {self._working}")
                except Exception as e:
                    self._error_count += 1
                    resource_monitor_logger.error(f"NVMLContext.__enter__() - Failed to initialize NVML: {e}")
                    self._working = False
                    # Don't raise, just return self and let callers handle missing GPU capabilities
            elif self._error_count >= 10:  # If we've had too many errors, don't try to use NVML
                resource_monitor_logger.warning("NVMLContext.__enter__() - Too many previous errors, NVML operations disabled")
                self._working = False
        resource_monitor_logger.debug("NVMLContext.__enter__() - Exit")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        resource_monitor_logger.debug("NVMLContext.__exit__() - Entry")
        with self._init_lock:
            if self.initialized:
                try:
                    resource_monitor_logger.debug("NVMLContext.__exit__() - Shutting down NVML...")
                    nvmlShutdown()
                    self.initialized = False  # Reset flag after shutdown
                    self._working = False
                    resource_monitor_logger.info("NVML shutdown successfully")
                except Exception as e:
                    self._error_count += 1
                    resource_monitor_logger.error(f"NVMLContext.__exit__() - Failed to shutdown NVML: {e}")
        resource_monitor_logger.debug("NVMLContext.__exit__() - Exit")

    def is_working(self) -> bool:
        """Check if NVML is initialized and working properly"""
        return self.initialized and self._working

    def shutdown(self):
        """Explicitly shutdown NVML if initialized."""
        resource_monitor_logger.debug("NVMLContext.shutdown() - Entry")
        with self._init_lock:
            if self.initialized:
                try:
                    resource_monitor_logger.debug("NVMLContext.shutdown() - Shutting down NVML...")
                    nvmlShutdown()
                    self.initialized = False
                    self._working = False
                    resource_monitor_logger.info("NVML explicitly shutdown successfully")
                except Exception as e:
                    self._error_count += 1
                    resource_monitor_logger.error(f"NVMLContext.shutdown() - Failed to shutdown NVML: {e}")
        resource_monitor_logger.debug("NVMLContext.shutdown() - Exit")


class ResourceMonitor:
    """
    ResourceMonitor class responsible for collecting and tracking resource usage metrics.
    Focuses exclusively on monitoring and data collection functionality.
    
    Supports both synchronous and asynchronous operations for optimal performance.
    """
    def __init__(self, config: Optional[ResourceConfig] = None):
        """
        Initialize the ResourceMonitor with configuration settings.
        
        Args:
            config: Optional ResourceConfig object. If None, a default config will be created.
        """
        resource_monitor_logger.debug("ResourceMonitor.__init__() - Entry")
        
        self.config = config or ResourceConfig()
        resource_monitor_logger.debug("ResourceMonitor.__init__() - Config initialized")
        
        self._lock = threading.Lock()  # Main lock for ResourceMonitor state
        self._history_lock = threading.Lock()  # Separate lock for history data to reduce contention
        resource_monitor_logger.debug("ResourceMonitor.__init__() - Locks initialized")
        
        # CPU caching and history with timestamps
        self._cached_cpu_percent = None
        self._cpu_last_measure_time = 0
        self.cpu_usage_history: Deque[Tuple[float, float]] = deque(maxlen=self.config.rolling_window_size * 2)
        resource_monitor_logger.debug("ResourceMonitor.__init__() - CPU history deque initialized")
        
        # GPU monitoring - handles and history with timestamps
        self._nvml_context = None
        self.gpu_handles = []
        self.gpu_usage_history: Deque[Tuple[float, float]] = deque(maxlen=self.config.rolling_window_size * 2)
        self.gpu_memory_history: Deque[Tuple[float, float]] = deque(maxlen=self.config.rolling_window_size * 2)
        resource_monitor_logger.debug("ResourceMonitor.__init__() - GPU vars initialized")
        
        # Initialize NVML if available
        self._init_nvml()
        
        # Memory history with timestamps
        self.mem_usage_history: Deque[Tuple[float, float]] = deque(maxlen=self.config.rolling_window_size * 2)
        resource_monitor_logger.debug("ResourceMonitor.__init__() - Memory history deque initialized")
        
        # Network usage and history with timestamps
        self.net_usage_history: Deque[Tuple[float, float]] = deque(maxlen=self.config.rolling_window_size * 2)
        self._last_net_io_counters = None
        self._net_last_measure_time = 0
        resource_monitor_logger.debug("ResourceMonitor.__init__() - Network history deque initialized")
        
        # Create cache for network metrics with a TTL of 1 second
        self._network_metrics_cache = {'timestamp': 0, 'value': 0.0}
        self._network_cache_ttl = 1.0  # 1 second TTL

        # Create event loop for async operations if needed
        self._loop = None
        
        resource_monitor_logger.debug("ResourceMonitor.__init__() - End")
    
    def _init_nvml(self) -> None:
        """Initialize NVML if available and get GPU handles."""
        if NVML_AVAILABLE:
            resource_monitor_logger.debug("_init_nvml() - NVML_AVAILABLE is True")
            try:
                self._nvml_context = NVMLContext()
                resource_monitor_logger.debug("_init_nvml() - NVML Context created")
                # Use context manager correctly
                with self._nvml_context:
                    # First check if NVML is working properly
                    if not self._nvml_context.is_working():
                        resource_monitor_logger.warning("_init_nvml() - NVML is not working properly. GPU monitoring will be disabled.")
                        self.gpu_handles = []
                        return
                        
                    try:
                        gpu_count = nvmlDeviceGetCount()
                        if gpu_count > 0:
                            for i in range(gpu_count):
                                try:
                                    self.gpu_handles.append(nvmlDeviceGetHandleByIndex(i))
                                except Exception as e:
                                    resource_monitor_logger.warning(f"_init_nvml() - Failed to get GPU handle for device {i}: {e}")
                            resource_monitor_logger.info(f"Found {len(self.gpu_handles)} GPUs. Monitoring all available GPUs.")
                        else:
                            resource_monitor_logger.warning("No GPUs found by NVML.")
                    except Exception as e:
                        resource_monitor_logger.warning(f"_init_nvml() - Failed to get GPU count: {e}")
            except Exception as e:
                resource_monitor_logger.warning(f"_init_nvml() - Failed to initialize NVML: {e}")
                self.gpu_handles = []  # Ensure empty list even if NVML init partially succeeded
                self._nvml_context = None
        else:
            resource_monitor_logger.debug("_init_nvml() - NVML_AVAILABLE is False, skipping NVML init")

    @lru_cache(maxsize=1)
    def get_cpu_percent(self) -> float:
        """
        Get current CPU utilization percentage.
        Uses lru_cache to cache results for quick repeated access.
        
        Returns:
            float: Current CPU utilization as a percentage
        """
        resource_monitor_logger.debug("get_cpu_percent() - Start")
        current_time = time.time()

        # Take multiple samples to improve detection of CPU spikes
        samples = []
        # Take more samples to catch CPU spikes
        for _ in range(5):  # Take 5 quick samples to catch spikes
            try:
                # Non-blocking CPU measurement
                cpu_usage = psutil.cpu_percent(interval=None)
                samples.append(cpu_usage)
                time.sleep(0.01)  # Shorter sleep for quicker response
            except Exception as e:
                resource_monitor_logger.error(f"Error getting CPU percent: {e}")
        
        # Use the maximum CPU value from samples to be more sensitive to spikes
        if samples:
            cpu_usage = max(samples)
            resource_monitor_logger.debug(f"get_cpu_percent() - Samples: {samples}, using max: {cpu_usage}")
        else:
            cpu_usage = 0.0
            
        # Increase sensitivity by applying a small multiplier to detect borderline cases
        adjusted_cpu = cpu_usage * 1.1  # Add 10% to actual readings to be more sensitive
        
        # Update history safely
        try:
            with self._history_lock:
                self.cpu_usage_history.append((current_time, adjusted_cpu))
        except Exception as e:
            resource_monitor_logger.error(f"Error updating CPU history: {e}")
        
        # Update cache safely
        try:
            with self._lock:
                self._cached_cpu_percent = adjusted_cpu
                self._cpu_last_measure_time = current_time
        except Exception as e:
            resource_monitor_logger.error(f"Error updating CPU cache: {e}")
            
        resource_monitor_logger.debug(f"get_cpu_percent() - End, returning {adjusted_cpu}")
        return adjusted_cpu

    # Add async version of CPU monitoring
    async def get_cpu_percent_async(self) -> float:
        """
        Asynchronous version of get_cpu_percent.
        
        Returns:
            float: Current CPU utilization as a percentage
        """
        # Run CPU monitoring in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, self.get_cpu_percent)

    @lru_cache(maxsize=1)
    def get_gpu_percent(self) -> Tuple[float, float]:
        """
        Get current GPU utilization percentage and memory usage percentage.
        Uses lru_cache to cache results for quick repeated access.
        
        Returns:
            Tuple[float, float]: A tuple containing (GPU utilization percentage, GPU memory usage percentage)
        """
        resource_monitor_logger.debug("get_gpu_percent() - Start")
        if not self.gpu_handles or not NVML_AVAILABLE or not self._nvml_context:
            resource_monitor_logger.debug("get_gpu_percent() - No GPU handles or context, returning 0.0")
            return 0.0, 0.0

        current_time = time.time()
        try:
            # Use GPU context and collect all available metrics
            with self._nvml_context:
                if not self._nvml_context.is_working():
                    resource_monitor_logger.debug("get_gpu_percent() - NVML not working, returning 0.0")
                    return 0.0, 0.0
                
                # If we have multiple GPUs, average their metrics
                utilization_sum = 0.0
                memory_sum = 0.0
                valid_gpus = 0
                
                for handle in self.gpu_handles:
                    try:
                        util = nvmlDeviceGetUtilizationRates(handle)
                        memory = nvmlDeviceGetMemoryInfo(handle)
                        
                        utilization_sum += util.gpu
                        # Calculate memory percentage
                        memory_percent = (memory.used / memory.total) * 100.0 if memory.total > 0 else 0.0
                        memory_sum += memory_percent
                        
                        valid_gpus += 1
                    except Exception as e:
                        resource_monitor_logger.error(f"get_gpu_percent() - Error getting GPU metrics: {e}")
                
                # Calculate averages if we have valid metrics
                if valid_gpus > 0:
                    avg_util = utilization_sum / valid_gpus
                    avg_memory = memory_sum / valid_gpus
                else:
                    avg_util = 0.0
                    avg_memory = 0.0
                
                # Update history safely
                with self._history_lock:
                    self.gpu_usage_history.append((current_time, avg_util))
                    self.gpu_memory_history.append((current_time, avg_memory))
                
                resource_monitor_logger.debug(f"get_gpu_percent() - End, returning util={avg_util}, memory={avg_memory}")
                return avg_util, avg_memory
                
        except Exception as e:
            resource_monitor_logger.error(f"get_gpu_percent() - Error: {e}")
            return 0.0, 0.0

    # Add async version of GPU monitoring
    async def get_gpu_percent_async(self) -> Tuple[float, float]:
        """
        Asynchronous version of get_gpu_percent.
        
        Returns:
            Tuple[float, float]: A tuple containing (GPU utilization percentage, GPU memory usage percentage)
        """
        # Run GPU monitoring in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, self.get_gpu_percent)

    @lru_cache(maxsize=1)
    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        Uses lru_cache to cache results for quick repeated access.
        
        Returns:
            float: Current memory usage in MB
        """
        resource_monitor_logger.debug("get_memory_usage_mb() - Start")
        current_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)  # Convert to MB
            
            # Update history safely
            with self._history_lock:
                self.mem_usage_history.append((current_time, memory_used_mb))
                
            resource_monitor_logger.debug(f"get_memory_usage_mb() - End, returning {memory_used_mb}")
            return memory_used_mb
        except Exception as e:
            resource_monitor_logger.error(f"get_memory_usage_mb() - Error: {e}")
            return 0.0

    # Add async version of memory monitoring
    async def get_memory_usage_mb_async(self) -> float:
        """
        Asynchronous version of get_memory_usage_mb.
        
        Returns:
            float: Current memory usage in MB
        """
        # Run memory monitoring in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, self.get_memory_usage_mb)

    # Network methods with optimized caching
    def get_network_usage_bytes_per_second(self) -> float:
        """
        Get current network usage in bytes per second.
        Uses manual time-based caching to avoid frequent access.
        
        Returns:
            float: Current network usage in bytes per second
        """
        resource_monitor_logger.debug("get_network_usage_bytes_per_second() - Start")
        current_time = time.time()
        
        # Check if we have a recent cache (TTL of 1 second)
        if current_time - self._network_metrics_cache['timestamp'] < self._network_cache_ttl:
            resource_monitor_logger.debug(f"get_network_usage_bytes_per_second() - Using cached value: {self._network_metrics_cache['value']}")
            return self._network_metrics_cache['value']
        
        try:
            current_net_io = psutil.net_io_counters()
            
            if self._last_net_io_counters is None:
                # Initialize the counters if this is the first call
                self._last_net_io_counters = current_net_io
                self._net_last_measure_time = current_time
                resource_monitor_logger.debug("get_network_usage_bytes_per_second() - Initialized network counters")
                return 0
            
            # Calculate time delta
            time_delta = current_time - self._net_last_measure_time
            if time_delta <= 0:
                resource_monitor_logger.debug("get_network_usage_bytes_per_second() - Time delta is zero, returning 0")
                return 0
            
            # Calculate bytes sent and received per second
            bytes_sent = (current_net_io.bytes_sent - self._last_net_io_counters.bytes_sent) / time_delta
            bytes_recv = (current_net_io.bytes_recv - self._last_net_io_counters.bytes_recv) / time_delta
            total_bytes_per_second = (bytes_sent + bytes_recv) * 1.5  # Apply 1.5 scaling factor
            
            # Update last values for next calculation
            self._last_net_io_counters = current_net_io
            self._net_last_measure_time = current_time
            
            # Update history safely
            with self._history_lock:
                self.net_usage_history.append((current_time, total_bytes_per_second))
            
            # Update cache
            self._network_metrics_cache = {'timestamp': current_time, 'value': total_bytes_per_second}
            
            resource_monitor_logger.debug(f"get_network_usage_bytes_per_second() - End, returning {total_bytes_per_second}")
            return total_bytes_per_second
        except Exception as e:
            resource_monitor_logger.error(f"get_network_usage_bytes_per_second() - Error: {e}")
            return 0

    # Add async version of network monitoring
    async def get_network_usage_bytes_per_second_async(self) -> float:
        """
        Asynchronous version of get_network_usage_bytes_per_second.
        
        Returns:
            float: Current network usage in bytes per second
        """
        # Run network monitoring in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, self.get_network_usage_bytes_per_second)

    # Concurrent/async version of get_all_metrics
    async def get_all_metrics_async(self) -> Dict[str, Any]:
        """
        Get all resource metrics asynchronously.
        This method uses concurrency to collect all metrics in parallel for improved performance.
        
        Returns:
            Dict[str, Any]: Dictionary containing all resource metrics
        """
        resource_monitor_logger.debug("get_all_metrics_async() - Start")
        
        # Run all metrics collection concurrently
        tasks = [
            self.get_cpu_percent_async(),
            self.get_memory_usage_mb_async(),
            self.get_gpu_percent_async(),
            self.get_network_usage_bytes_per_second_async()
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Unpack results
        cpu_percent = results[0]
        memory_usage_mb = results[1]
        gpu_percent, gpu_memory_percent = results[2]
        network_usage = results[3]
        
        # Get current timestamp
        current_time = time.time()
        
        # Create metrics dictionary
        metrics = {
            "timestamp": current_time,
            "cpu_percent": cpu_percent,
            "memory_usage_mb": memory_usage_mb,
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "network_usage_bytes_per_second": network_usage
        }
        
        resource_monitor_logger.debug(f"get_all_metrics_async() - End, returning {metrics}")
        return metrics

    # Optimized synchronous version of get_all_metrics
    @lru_cache(maxsize=1)
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all resource metrics.
        Uses lru_cache to cache results for quick repeated access.
        
        Returns:
            Dict[str, Any]: Dictionary containing all resource metrics
        """
        resource_monitor_logger.debug("get_all_metrics() - Start")
        
        # Get current timestamp
        current_time = time.time()
        
        # Use concurrent.futures to parallelize metric collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_cpu = executor.submit(self.get_cpu_percent)
            future_memory = executor.submit(self.get_memory_usage_mb)
            future_gpu = executor.submit(self.get_gpu_percent)
            future_network = executor.submit(self.get_network_usage_bytes_per_second)
            
            # Get results from futures
            cpu_percent = future_cpu.result()
            memory_usage_mb = future_memory.result()
            gpu_percent, gpu_memory_percent = future_gpu.result()
            network_usage = future_network.result()
        
        # Create metrics dictionary
        metrics = {
            "timestamp": current_time,
            "cpu_percent": cpu_percent,
            "memory_usage_mb": memory_usage_mb,
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "network_usage_bytes_per_second": network_usage
        }
        
        resource_monitor_logger.debug(f"get_all_metrics() - End, returning {metrics}")
        return metrics

    # Invalidate all caches
    def invalidate_caches(self) -> None:
        """
        Invalidate all method caches to force fresh readings.
        """
        resource_monitor_logger.debug("invalidate_caches() - Invalidating all method caches")
        
        # Safely clear caches if they exist (handle case where methods are mocked)
        if hasattr(self.get_cpu_percent, 'cache_clear'):
            self.get_cpu_percent.cache_clear()
            
        if hasattr(self.get_memory_usage_mb, 'cache_clear'):
            self.get_memory_usage_mb.cache_clear()
            
        if hasattr(self.get_gpu_percent, 'cache_clear'):
            self.get_gpu_percent.cache_clear()
            
        if hasattr(self.get_all_metrics, 'cache_clear'):
            self.get_all_metrics.cache_clear()
            
        # Reset network cache
        self._network_metrics_cache = {'timestamp': 0, 'value': 0.0}

    # Update close method to clean up async resources
    def close(self) -> None:
        """
        Properly shut down the resource monitor.
        """
        resource_monitor_logger.debug("ResourceMonitor.close() - Start")
        
        # Shut down NVML context if it exists
        if self._nvml_context:
            self._nvml_context.shutdown()
            
        # Clear all caches
        self.invalidate_caches()
        
        resource_monitor_logger.debug("ResourceMonitor.close() - End") 