import time
import logging
import threading
import asyncio
import concurrent.futures
from functools import lru_cache
from typing import Optional, List, Tuple, Dict
from .resource_config import ResourceConfig
from .resource_monitor import ResourceMonitor

resource_logger = logging.getLogger("resource_manager")

# Create thread pool executor for CPU-bound tasks
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

class ResourceManager:
    """
    ResourceManager class responsible for analyzing resource data and making pruning decisions.
    Uses ResourceMonitor to obtain resource usage data.
    
    Supports both synchronous and asynchronous operations for optimal performance.
    """
    def __init__(self,
                 config: Optional[ResourceConfig] = None,
                 max_memory_usage_mb: Optional[float] = None,
                 max_cpu_usage_percent: Optional[float] = None,
                 max_gpu_usage_percent: Optional[float] = None,
                 max_network_usage_bytes_per_second: Optional[float] = None,
                 check_interval: Optional[float] = None,
                 resource_monitor: Optional[ResourceMonitor] = None):
        """
        Initialize the ResourceManager with configuration settings.
        
        Args:
            config: Optional ResourceConfig object. If None, a default config will be created.
            max_memory_usage_mb: Optional override for max memory usage threshold.
            max_cpu_usage_percent: Optional override for max CPU usage threshold.
            max_gpu_usage_percent: Optional override for max GPU usage threshold.
            max_network_usage_bytes_per_second: Optional override for max network usage threshold.
            check_interval: Optional override for check interval.
            resource_monitor: Optional ResourceMonitor instance. If None, a new one will be created.
        """
        resource_logger.debug("ResourceManager.__init__() - Entry - Log Point 1")  # Log Point 1: Entry to init

        self.config = config or ResourceConfig()
        resource_logger.debug("ResourceManager.__init__() - Config initialized - Log Point 2")  # Log Point 2: Config init done

        self.observation_period = self.config.observation_period
        self.target_utilization = self.config.target_utilization
        self.check_interval = check_interval if check_interval is not None else self.config.check_interval
        resource_logger.debug("ResourceManager.__init__() - Check interval set - Log Point 3")  # Log Point 3: Check interval set
        if self.check_interval <= 0:
            raise ValueError("Check interval must be positive.")
        resource_logger.debug("ResourceManager.__init__() - Check interval validated - Log Point 4")  # Log Point 4: Check interval validated

        self.max_memory_usage_mb = max_memory_usage_mb if max_memory_usage_mb is not None else self.config.max_memory_usage_mb
        self.max_cpu_usage_percent = max_cpu_usage_percent if max_cpu_usage_percent is not None else self.config.max_cpu_usage_percent
        self.max_gpu_usage_percent = max_gpu_usage_percent if max_gpu_usage_percent is not None else self.config.max_gpu_usage_percent
        self.max_network_usage_bytes_per_second = max_network_usage_bytes_per_second if max_network_usage_bytes_per_second is not None else self.config.max_network_usage_bytes_per_second
        resource_logger.debug("ResourceManager.__init__() - Max resource limits set - Log Point 5")  # Log Point 5: Max limits set

        self.last_check = time.time()
        self._lock = threading.Lock()  # Main lock for ResourceManager state
        # Add asyncio lock for async methods
        self._async_lock = asyncio.Lock()
        resource_logger.debug("ResourceManager.__init__() - Locks initialized - Log Point 6")  # Log Point 6: Lock init

        # Initialize ResourceMonitor
        self.resource_monitor = resource_monitor or ResourceMonitor(config=self.config)
        resource_logger.debug("ResourceManager.__init__() - ResourceMonitor initialized")

        # Pre-filter history lists for analysis
        self._filtered_cpu_history: List[Tuple[float, float]] = []
        self._filtered_mem_history: List[Tuple[float, float]] = []
        self._filtered_gpu_history: List[Tuple[float, float]] = []
        self._filtered_net_history: List[Tuple[float, float]] = []
        self._last_filter_time = 0
        
        # Cache for pruning decisions with TTL
        self._prune_decision_cache = {'timestamp': 0, 'decision': False}
        self._prune_cache_ttl = self.check_interval * 0.9  # 90% of check interval

        resource_logger.debug("ResourceManager.__init__() - End")  # Log exit from init

    def _update_filtered_histories(self, current_time: float) -> None:
        """
        Filter resource histories to only include entries within observation period.
        
        Args:
            current_time: Current timestamp to use as reference point
        """
        # Only filter histories if it's been a while since the last filtering
        if current_time - self._last_filter_time < self.check_interval:
            return
            
        cutoff_time = current_time - self.observation_period
        
        with self.resource_monitor._history_lock:
            # Filter CPU history
            self._filtered_cpu_history = [(t, v) for t, v in self.resource_monitor.cpu_usage_history if t >= cutoff_time]
            
            # Filter memory history
            self._filtered_mem_history = [(t, v) for t, v in self.resource_monitor.mem_usage_history if t >= cutoff_time]
            
            # Filter GPU history
            self._filtered_gpu_history = [(t, v) for t, v in self.resource_monitor.gpu_usage_history if t >= cutoff_time]
            
            # Filter network history
            self._filtered_net_history = [(t, v) for t, v in self.resource_monitor.net_usage_history if t >= cutoff_time]
        
        self._last_filter_time = current_time

    # Optimized version with lru_cache for better performance
    @lru_cache(maxsize=8)
    def should_prune(self) -> bool:
        """
        Determine if resource pruning is necessary based on current resource usage.
        Uses caching for better performance.
        
        This method checks if the average resource utilization over the observation period
        exceeds the target utilization AND if current resource usage exceeds maximum thresholds.
        
        Returns:
            bool: True if pruning is necessary, False otherwise
        """
        current_time = time.time()
        
        # First check if cached decision is still valid
        if current_time - self._prune_decision_cache['timestamp'] < self._prune_cache_ttl:
            resource_logger.debug(f"should_prune() - Using cached decision: {self._prune_decision_cache['decision']}")
            return self._prune_decision_cache['decision']
        
        if not self.config.enabled:
            resource_logger.debug("should_prune() - Resource manager disabled, returning False")
            return False
        
        # Check if enough time has passed since last check
        with self._lock:
            if current_time - self.last_check < self.check_interval:
                resource_logger.debug(f"should_prune() - Skipping check, interval not reached. Last: {self.last_check}, Current: {current_time}")
                return False
            self.last_check = current_time
        
        # Collect current resource metrics using concurrent execution
        resource_metrics = self._get_current_resource_metrics()
        
        # Filter histories to observation period
        self._update_filtered_histories(current_time)
        
        # Use concurrent.futures to parallelize resource condition checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_cpu = executor.submit(
                self._check_resource_condition,
                resource_metrics["cpu_percent"],
                self._filtered_cpu_history,
                self.max_cpu_usage_percent,
                "CPU"
            )
            
            future_memory = executor.submit(
                self._check_resource_condition,
                resource_metrics["memory_usage_mb"],
                self._filtered_mem_history,
                self.max_memory_usage_mb,
                "Memory"
            )
            
            # Only check GPU if being used
            if resource_metrics["gpu_percent"] > 0:
                future_gpu = executor.submit(
                    self._check_resource_condition,
                    resource_metrics["gpu_percent"],
                    self._filtered_gpu_history,
                    self.max_gpu_usage_percent,
                    "GPU"
                )
            else:
                future_gpu = None
            
            future_network = executor.submit(
                self._check_resource_condition,
                resource_metrics["network_usage_bytes_per_second"],
                self._filtered_net_history,
                self.max_network_usage_bytes_per_second,
                "Network"
            )
            
            # Get results from futures
            cpu_should_prune = future_cpu.result()
            memory_should_prune = future_memory.result()
            gpu_should_prune = future_gpu.result() if future_gpu else False
            network_should_prune = future_network.result()
        
        # Final pruning decision - if any resource needs pruning
        should_prune = any([cpu_should_prune, memory_should_prune, gpu_should_prune, network_should_prune])
        
        resource_logger.info(f"Resource pruning decision: {should_prune} (CPU: {cpu_should_prune}, Memory: {memory_should_prune}, GPU: {gpu_should_prune}, Network: {network_should_prune})")
        
        # Cache the pruning decision
        self._prune_decision_cache = {'timestamp': current_time, 'decision': should_prune}
        
        return should_prune

    # Add asynchronous version of should_prune
    async def should_prune_async(self) -> bool:
        """
        Asynchronous version of should_prune.
        
        Returns:
            bool: True if pruning is necessary, False otherwise
        """
        # First check if cached decision is still valid
        current_time = time.time()
        if current_time - self._prune_decision_cache['timestamp'] < self._prune_cache_ttl:
            resource_logger.debug(f"should_prune_async() - Using cached decision: {self._prune_decision_cache['decision']}")
            return self._prune_decision_cache['decision']
        
        if not self.config.enabled:
            resource_logger.debug("should_prune_async() - Resource manager disabled, returning False")
            return False
        
        # Check if enough time has passed since last check
        async with self._async_lock:
            if current_time - self.last_check < self.check_interval:
                resource_logger.debug(f"should_prune_async() - Skipping check, interval not reached. Last: {self.last_check}, Current: {current_time}")
                return False
            self.last_check = current_time
        
        # Get metrics asynchronously
        resource_metrics = await self._get_current_resource_metrics_async()
        
        # Filter histories
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_thread_pool, self._update_filtered_histories, current_time)
        
        # Create async tasks for resource condition checks
        tasks = []
        
        # CPU task
        tasks.append(self._check_resource_condition_async(
            resource_metrics["cpu_percent"],
            self._filtered_cpu_history,
            self.max_cpu_usage_percent,
            "CPU"
        ))
        
        # Memory task
        tasks.append(self._check_resource_condition_async(
            resource_metrics["memory_usage_mb"],
            self._filtered_mem_history,
            self.max_memory_usage_mb,
            "Memory"
        ))
        
        # GPU task (only if GPU is being used)
        if resource_metrics["gpu_percent"] > 0:
            tasks.append(self._check_resource_condition_async(
                resource_metrics["gpu_percent"],
                self._filtered_gpu_history,
                self.max_gpu_usage_percent,
                "GPU"
            ))
        
        # Network task
        tasks.append(self._check_resource_condition_async(
            resource_metrics["network_usage_bytes_per_second"],
            self._filtered_net_history,
            self.max_network_usage_bytes_per_second,
            "Network"
        ))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Final pruning decision - if any resource needs pruning
        should_prune = any(results)
        
        resource_logger.info(f"Resource pruning decision (async): {should_prune}")
        
        # Cache the pruning decision
        self._prune_decision_cache = {'timestamp': current_time, 'decision': should_prune}
        
        return should_prune

    def _get_current_resource_metrics(self) -> Dict[str, float]:
        """
        Get current resource metrics from ResourceMonitor.
        
        Returns:
            Dict[str, float]: Dictionary of current resource metrics
        """
        metrics = self.resource_monitor.get_all_metrics()
        return metrics

    # Add asynchronous version of resource metrics collection
    async def _get_current_resource_metrics_async(self) -> Dict[str, float]:
        """
        Get current resource metrics from ResourceMonitor asynchronously.
        
        Returns:
            Dict[str, float]: Dictionary of current resource metrics
        """
        metrics = await self.resource_monitor.get_all_metrics_async()
        return metrics

    def _check_resource_condition(self, current_value: float, history: List[Tuple[float, float]], 
                                 max_threshold: float, resource_name: str) -> bool:
        """
        Check if a specific resource needs pruning based on current value and history.
        
        Args:
            current_value: Current resource usage value
            history: List of (timestamp, value) tuples representing usage history
            max_threshold: Maximum allowed threshold for this resource
            resource_name: Name of the resource for logging purposes
            
        Returns:
            bool: True if this resource needs pruning, False otherwise
        """
        # Check if current value exceeds max threshold
        if current_value > max_threshold:
            # Calculate average utilization over observation period
            if history:
                values_only = [value for _, value in history]
                avg_value = sum(values_only) / len(values_only)
                utilization_ratio = avg_value / max_threshold
                
                # Check if average utilization exceeds target utilization
                if utilization_ratio > self.target_utilization:
                    resource_logger.warning(
                        f"{resource_name} utilization ({current_value:.2f}) exceeds max threshold ({max_threshold:.2f}) "
                        f"and average utilization ({avg_value:.2f}, {utilization_ratio:.2%}) exceeds target ({self.target_utilization:.2%})"
                    )
                    return True
                else:
                    resource_logger.debug(
                        f"{resource_name} utilization ({current_value:.2f}) exceeds max threshold ({max_threshold:.2f}) "
                        f"but average utilization ({avg_value:.2f}, {utilization_ratio:.2%}) is below target ({self.target_utilization:.2%})"
                    )
            else:
                resource_logger.debug(
                    f"{resource_name} utilization ({current_value:.2f}) exceeds max threshold ({max_threshold:.2f}) "
                    f"but no history available for average calculation"
                )
        
        return False

    # Add asynchronous version of resource condition checking
    async def _check_resource_condition_async(self, current_value: float, history: List[Tuple[float, float]], 
                                     max_threshold: float, resource_name: str) -> bool:
        """
        Asynchronous version of _check_resource_condition.
        
        Args:
            current_value: Current resource usage value
            history: List of (timestamp, value) tuples representing usage history
            max_threshold: Maximum allowed threshold for this resource
            resource_name: Name of the resource for logging purposes
            
        Returns:
            bool: True if this resource needs pruning, False otherwise
        """
        # Run in thread pool to avoid blocking the event loop with CPU-bound calculations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _thread_pool, 
            self._check_resource_condition, 
            current_value, 
            history, 
            max_threshold, 
            resource_name
        )

    # Clear the pruning decision cache
    def invalidate_cache(self) -> None:
        """
        Invalidate the pruning decision cache to force fresh evaluation.
        """
        resource_logger.debug("invalidate_cache() - Invalidating pruning decision cache")
        self._prune_decision_cache = {'timestamp': 0, 'decision': False}
        self.should_prune.cache_clear()
        
        # Also invalidate ResourceMonitor caches
        if self.resource_monitor:
            self.resource_monitor.invalidate_caches()

    def close(self) -> None:
        """
        Properly shut down the resource manager and its monitor.
        """
        resource_logger.debug("ResourceManager.close() - Start")
        
        # Clear caches
        self.invalidate_cache()
        
        # Close the resource monitor
        if self.resource_monitor:
            self.resource_monitor.close()
        
        resource_logger.debug("ResourceManager.close() - End")