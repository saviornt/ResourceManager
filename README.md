# ResourceManager

A Python utility for monitoring and managing system resources. It helps prevent system overload by monitoring CPU, GPU, memory, and network usage while providing an API for resource status and pruning decisions.

## Features

* **Real-time Resource Monitoring:** Track CPU, GPU, memory, and network usage in real-time.
* **Adaptive Resource Management:** Define thresholds for resource usage and get pruning signals when they're exceeded.
* **Asynchronous Operation:** All operations can be performed asynchronously.
* **FastAPI Integration:** Built-in REST API for remote monitoring and management.
* **Simple Integration:** Designed to be easily dropped into existing Python projects.
* **Python 3.12+ and CUDA Compatible:** Supports Python 3.12 and seamlessly integrates with CUDA environments when NVIDIA GPUs and `pynvml` are available.
* **Comprehensive Error Handling and Resilience:** Includes proper error handling and resilience in API endpoints.
* **Extensive Test Suite with High Coverage:** Includes a test suite with high coverage to ensure reliability.

## Installation

```bash
pip install git+https://github.com/saviornt/resourcemanager
```

## Usage

```python
from src.resource_manager import ResourceManager

# Initialize with custom thresholds
manager = ResourceManager(
    max_cpu_percent=80.0,
    max_memory_percent=75.0,
    max_gpu_percent=90.0,
    max_network_bytes_per_sec=1_000_000
)

# Check if pruning is needed
if manager.should_prune():
    print("Pruning recommended due to high resource usage")
    
# Get individual metrics
cpu_usage = manager.resource_monitor.get_cpu_percent()
memory_usage = manager.resource_monitor.get_memory_usage_mb()

# Asynchronous usage
import asyncio

async def check_resources():
    should_prune = await manager.should_prune_async()
    if should_prune:
        print("Async pruning recommended")
        
# Close when done
manager.close()
```

## GitHub Repository

<https://github.com/saviornt/ResourceManager>

## API Reference

ResourceManager provides a FastAPI-based REST API for remote monitoring and resource management.

### Server Setup

To start the API server:

```python
import uvicorn
from resource_manager import endpoints

if __name__ == "__main__":
    uvicorn.run(endpoints, host="0.0.0.0", port=8000)
```

### Endpoints

The API provides the following endpoints:

#### 1. `/endpoint/cpu`

**Method:** GET  
**Response Model:** `CPUResponse`  
**Description:** Returns current CPU usage percentage.

```json
{
  "cpu_percent": 25.7,
  "timestamp": 1625482956.7891
}
```

#### 2. `/endpoint/memory`

**Method:** GET  
**Response Model:** `MemoryResponse`  
**Description:** Returns current memory usage in MB.

```json
{
  "memory_usage_mb": 4852.6,
  "timestamp": 1625482956.7891
}
```

#### 3. `/endpoint/gpu`

**Method:** GET  
**Response Model:** `GPUResponse`  
**Description:** Returns current GPU usage and memory usage percentages.

```json
{
  "gpu_percent": 42.3,
  "gpu_memory_percent": 35.8,
  "timestamp": 1625482956.7891
}
```

#### 4. `/endpoint/network`

**Method:** GET  
**Response Model:** `NetworkResponse`  
**Description:** Returns current network usage in bytes per second.

```json
{
  "network_usage_bytes_per_second": 1250000,
  "timestamp": 1625482956.7891
}
```

#### 5. `/endpoint/all`

**Method:** GET  
**Response Model:** `AllResourcesResponse`  
**Description:** Returns all resource metrics in a single response.

```json
{
  "cpu_percent": 25.7,
  "memory_usage_mb": 4852.6,
  "gpu_percent": 42.3,
  "gpu_memory_percent": 35.8,
  "network_usage_bytes_per_second": 1250000,
  "timestamp": 1625482956.7891
}
```

#### 6. `/endpoint/prune`

**Method:** GET  
**Response Model:** `PruneResponse`  
**Description:** Returns a boolean indicating whether resource pruning is necessary based on configured thresholds.

```json
{
  "should_prune": true,
  "timestamp": 1625482956.7891
}
```

### Response Models

The API uses the following Pydantic models for responses:

* `CPUResponse`: CPU utilization data
* `MemoryResponse`: Memory usage data
* `GPUResponse`: GPU utilization and memory data
* `NetworkResponse`: Network usage data
* `AllResourcesResponse`: Combined resource data
* `PruneResponse`: Pruning decision data

### API Features

* **Caching**: All responses are cached for 1 second to reduce system load
* **Background Tasks**: Cache is automatically refreshed in the background
* **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
* **Clean Shutdown**: Resources are properly cleaned up when the API server shuts down
