from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from typing import Optional, Dict, Any
import logging
import functools
import time
from pydantic import BaseModel, Field
from .resource_monitor import ResourceMonitor
from .resource_manager import ResourceManager
from contextlib import asynccontextmanager

# Set up logger
api_logger = logging.getLogger("resource_api")

# Shared cache for API responses
_api_cache: Dict[str, Dict[str, Any]] = {}

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing special needed
    yield
    # Shutdown: clean up resources
    global _resource_monitor, _resource_manager
    if _resource_monitor:
        api_logger.info("Shutting down ResourceMonitor")
        _resource_monitor.close()
        _resource_monitor = None
    if _resource_manager:
        api_logger.info("Shutting down ResourceManager")
        _resource_manager.close()
        _resource_manager = None

# Create FastAPI app
app = FastAPI(
    title="Resource Monitor API",
    description="API for real-time system resource monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Shared instances
_resource_monitor: Optional[ResourceMonitor] = None
_resource_manager: Optional[ResourceManager] = None

# Function to invalidate all caches
def invalidate_all_caches():
    """
    Invalidate all API response caches.
    """
    global _api_cache
    api_logger.debug("Invalidating all API response caches")
    _api_cache.clear()

# Background refresh task for caches
async def refresh_caches(background_tasks: BackgroundTasks):
    """
    Background task to refresh caches periodically.
    """
    if _resource_monitor:
        _resource_monitor.invalidate_caches()
    if _resource_manager:
        _resource_manager.invalidate_cache()
    
    # Also invalidate API caches
    invalidate_all_caches()

def get_resource_monitor() -> ResourceMonitor:
    """
    Factory function to get or create a ResourceMonitor instance.
    This function is used as a FastAPI dependency.
    
    Returns:
        ResourceMonitor: A shared ResourceMonitor instance
    """
    global _resource_monitor
    if _resource_monitor is None:
        api_logger.info("Initializing ResourceMonitor for API")
        _resource_monitor = ResourceMonitor()
    return _resource_monitor

def get_resource_manager() -> ResourceManager:
    """
    Factory function to get or create a ResourceManager instance.
    This function is used as a FastAPI dependency.
    
    Returns:
        ResourceManager: A shared ResourceManager instance
    """
    global _resource_manager, _resource_monitor
    if _resource_manager is None:
        api_logger.info("Initializing ResourceManager for API")
        _resource_manager = ResourceManager(resource_monitor=get_resource_monitor())
    return _resource_manager

# Function to cache API responses
def cache_response(func):
    """
    Decorator to cache API responses.
    """
    global _api_cache
    ttl = 1.0  # 1 second TTL
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get current timestamp and check cache
        current_time = time.time()
        
        # Use function name as cache key
        cache_key = func.__name__
        
        # Check if we have a valid cache
        if cache_key in _api_cache and current_time - _api_cache[cache_key]['timestamp'] < ttl:
            return _api_cache[cache_key]['data']
        
        # No valid cache, call original function
        try:
            result = await func(*args, **kwargs)
        
            # Cache result
            _api_cache[cache_key] = {
                'timestamp': current_time,
                'data': result
            }
            
            return result
        except Exception as e:
            api_logger.error(f"Error in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching resource data: {str(e)}")
    
    # Add method to invalidate cache for this function
    wrapper.invalidate_cache = lambda: _api_cache.pop(func.__name__, None)
    
    return wrapper


# Response models
class CPUResponse(BaseModel):
    """CPU resource data response model."""
    cpu_percent: float = Field(..., description="CPU utilization percentage")
    timestamp: float = Field(..., description="Timestamp of the measurement")


class MemoryResponse(BaseModel):
    """Memory resource data response model."""
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    timestamp: float = Field(..., description="Timestamp of the measurement")


class GPUResponse(BaseModel):
    """GPU resource data response model."""
    gpu_percent: float = Field(..., description="GPU utilization percentage")
    gpu_memory_percent: float = Field(..., description="GPU memory usage percentage")
    timestamp: float = Field(..., description="Timestamp of the measurement")


class NetworkResponse(BaseModel):
    """Network resource data response model."""
    network_usage_bytes_per_second: float = Field(..., description="Network usage in bytes per second")
    timestamp: float = Field(..., description="Timestamp of the measurement")


class AllResourcesResponse(BaseModel):
    """Combined resource data response model."""
    cpu_percent: float = Field(..., description="CPU utilization percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    gpu_percent: float = Field(..., description="GPU utilization percentage")
    gpu_memory_percent: float = Field(..., description="GPU memory usage percentage")
    network_usage_bytes_per_second: float = Field(..., description="Network usage in bytes per second")
    timestamp: float = Field(..., description="Timestamp of the measurement")


class PruneResponse(BaseModel):
    """Pruning decision response model."""
    should_prune: bool = Field(..., description="Whether pruning is necessary")
    timestamp: float = Field(..., description="Timestamp of the decision")


# API routes - Updated with /endpoint prefix
@app.get("/endpoint/cpu", response_model=CPUResponse)
@cache_response
async def get_cpu_usage(
    background_tasks: BackgroundTasks,
    monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Get current CPU usage percentage.
    """
    try:
        cpu_percent = await monitor.get_cpu_percent_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        return {
            "cpu_percent": cpu_percent,
            "timestamp": monitor._cpu_last_measure_time
        }
    except Exception as e:
        api_logger.error(f"Error getting CPU usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching CPU data: {str(e)}")


@app.get("/endpoint/memory", response_model=MemoryResponse)
@cache_response
async def get_memory_usage(
    background_tasks: BackgroundTasks,
    monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Get current memory usage in MB.
    """
    try:
        memory_usage_mb = await monitor.get_memory_usage_mb_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        return {
            "memory_usage_mb": memory_usage_mb,
            "timestamp": monitor._cpu_last_measure_time  # Reusing last timestamp as approximation
        }
    except Exception as e:
        api_logger.error(f"Error getting memory usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching memory data: {str(e)}")


@app.get("/endpoint/gpu", response_model=GPUResponse)
@cache_response
async def get_gpu_usage(
    background_tasks: BackgroundTasks,
    monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Get current GPU usage percentages.
    """
    try:
        gpu_percent, gpu_memory_percent = await monitor.get_gpu_percent_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        return {
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "timestamp": monitor._cpu_last_measure_time  # Reusing last timestamp as approximation
        }
    except Exception as e:
        api_logger.error(f"Error getting GPU usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching GPU data: {str(e)}")


@app.get("/endpoint/network", response_model=NetworkResponse)
@cache_response
async def get_network_usage(
    background_tasks: BackgroundTasks,
    monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Get current network usage in bytes per second.
    """
    try:
        network_usage = await monitor.get_network_usage_bytes_per_second_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        return {
            "network_usage_bytes_per_second": network_usage,
            "timestamp": monitor._net_last_measure_time
        }
    except Exception as e:
        api_logger.error(f"Error getting network usage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching network data: {str(e)}")


@app.get("/endpoint/all", response_model=AllResourcesResponse)
@cache_response
async def get_all_resources(
    background_tasks: BackgroundTasks,
    monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Get all resource metrics in a single response.
    """
    try:
        metrics = await monitor.get_all_metrics_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        return metrics
    except Exception as e:
        api_logger.error(f"Error getting all resource metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching resource data: {str(e)}")

@app.get("/endpoint/prune", response_model=PruneResponse)
@cache_response
async def should_prune(
    background_tasks: BackgroundTasks,
    manager: ResourceManager = Depends(get_resource_manager)
):
    """
    Check if resource pruning is necessary.
    """
    try:
        should_prune_result = await manager.should_prune_async()
        
        # Schedule a background task to refresh caches periodically
        background_tasks.add_task(refresh_caches, background_tasks)
        
        # Get current timestamp
        current_time = time.time()
        
        return {
            "should_prune": should_prune_result,
            "timestamp": current_time
        }
    except Exception as e:
        api_logger.error(f"Error checking if pruning is necessary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking pruning status: {str(e)}")

# Export the app as 'endpoints' to match the import in __init__.py
endpoints = app 