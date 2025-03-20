# ResourceManager

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## A Dynamic Resource Monitor and Manager for Python Applications

ResourceManager is a Python library designed to automatically monitor and manage system resources (CPU, Memory, GPU, Network) for your applications. It helps in dynamically detecting resource over-utilization and can be used as a trigger to implement resource management strategies within your applications, such as pruning less important tasks, reducing batch sizes, or offloading processes.

## Key Features

* **Modular Architecture:** Separated into ResourceMonitor (data collection) and ResourceManager (analysis and decision-making) components for better code organization.
* **REST API:** Real-time resource monitoring API endpoints for integration with external tools and dashboards.
* **Dynamic Resource Monitoring:** Tracks CPU, memory, GPU (NVIDIA GPUs via NVML), and network usage.
* **Configurable Limits:** Set maximum resource usage thresholds for memory, CPU, GPU, and network bandwidth.
* **Intelligent Pruning Decisions:** Utilizes an observation period and target utilization to make informed decisions about resource management, avoiding unnecessary actions based on short spikes.
* **Rolling Averages:** Smooths resource readings for more stable monitoring.
* **Multi-GPU Support:** Monitors all available NVIDIA GPUs in a system.
* **Thread-Safe Design:** Safe to use in multithreaded Python applications.
* **Easy Configuration:** Configuration through environment variables or a `ResourceConfig` object, leveraging `.env` files.
* **Simple Integration:** Designed to be easily dropped into existing Python projects.
* **Python 3.12+ and CUDA Compatible:** Supports Python 3.12 and seamlessly integrates with CUDA environments when NVIDIA GPUs and `pynvml` are available.

## Installation

You can install ResourceManager directly from GitHub using pip:

```bash
pip install git+https://github.com/saviornt/ResourceManager.git#egg=ResourceManager
```

**Note:**

* This command will install the ResourceManager package and its dependencies.
* If you intend to use GPU monitoring, ensure you have NVIDIA drivers and the pynvml library correctly installed in your environment. pynvml usually requires NVIDIA CUDA Toolkit to be installed.
* For network bandwidth monitoring, the library estimates usage based on bytes sent and received per second.

**Dependencies:**

* psutil>=7.0.0
* pydantic>=2.10.6
* pynvml>=12.0.0 (Optional, for GPU monitoring)
* numpy>=2.2.4
* numba>=0.61.0
* fastapi>=0.110.3 (For API endpoints)
* uvicorn>=0.29.0 (For running the API server)

## Basic Usage Example

```python
import logging
from src.resource_manager import ResourceManager
from src.resource_config import ResourceConfig
import time

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)
resource_logger = logging.getLogger("resource_manager")

# Initialize ResourceManager with default configuration (or customize via .env or ResourceConfig)
resource_manager = ResourceManager()

# Or, create a custom ResourceConfig object
custom_config = ResourceConfig(
    observation_period=20.0,
    check_interval=2.0,
    target_utilization=0.85,
    max_memory_usage_mb=4096.0,
    max_cpu_usage_percent=80.0,
    max_gpu_usage_percent=75.0,
    max_network_usage_bytes_per_second=5 * 1024 * 1024.0 # 5MB/s
)
# resource_manager = ResourceManager(config=custom_config) # Use custom config

try:
    while True:
        if resource_manager.should_prune():
            resource_logger.warning("Resource usage exceeds limits! Pruning action should be taken.")
            # Implement your resource pruning logic here
            # For example: reduce batch size, offload tasks, etc.
        else:
            resource_logger.info("Resource usage within limits.")

        time.sleep(resource_manager.check_interval)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    resource_manager.close() # Important to shutdown NVML if initialized
```

## Using ResourceMonitor Directly

If you only want to monitor resources without the pruning logic:

```python
from src.resource_monitor import ResourceMonitor
import time

# Initialize ResourceMonitor
monitor = ResourceMonitor()

try:
    # Get individual metrics
    cpu_percent = monitor.get_cpu_percent()
    memory_mb = monitor.get_memory_usage_mb()
    gpu_percent, gpu_memory_percent = monitor.get_gpu_percent()
    network_bps = monitor.get_network_usage_bytes_per_second()
    
    print(f"CPU: {cpu_percent:.2f}%, Memory: {memory_mb:.2f}MB")
    print(f"GPU: {gpu_percent:.2f}%, GPU Memory: {gpu_memory_percent:.2f}%")
    print(f"Network: {network_bps/1024/1024:.2f}MB/s")
    
    # Or get all metrics at once
    all_metrics = monitor.get_all_metrics()
    print(f"All metrics: {all_metrics}")
    
finally:
    monitor.close()
```

## Running the API Server

To start the API server for real-time resource monitoring:

```python
import uvicorn
from src.api import app

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## API Endpoints

The following API endpoints are available:

* `GET /resources/cpu` - Get CPU usage data
* `GET /resources/memory` - Get memory usage data
* `GET /resources/gpu` - Get GPU usage data
* `GET /resources/network` - Get network usage data
* `GET /resources/all` - Get all resource metrics in a single response

Example API response from `/resources/all`:

```json
{
  "cpu_percent": 23.5,
  "memory_usage_mb": 4096.2,
  "gpu_percent": 45.7,
  "gpu_memory_percent": 32.1,
  "network_usage_bytes_per_second": 2097152.0,
  "timestamp": 1620000000.123
}
```

## Configuration

ResourceManager can be configured in two primary ways:

1. Environment Variables: Set environment variables to override default configuration values.
2. ResourceConfig Object: Programmatically create a ResourceConfig object and pass it to the ResourceManager constructor. This allows for more dynamic or code-driven configuration.

You can store these in a .env file in the same directory as your script, and ResourceManager will automatically load them.
Example environment variables include:

```bash
OBSERVATION_PERIOD=15.0
CHECK_INTERVAL=1.5
TARGET_UTILIZATION=0.8
MAX_MEMORY_USAGE_MB=6144.0
MAX_CPU_USAGE_PERCENT=70.0
MAX_GPU_USAGE_PERCENT=65.0
MAX_NETWORK_USAGE_BYTES_PER_SECOND=8388608.0 # 8MB/s
```

## Important Notes

* **Modular Architecture:** The codebase is now separated into `resource_monitor.py` (data collection) and `resource_manager.py` (analysis and decision making) for better maintainability.

* **Pruning Logic is Application-Specific:** ResourceManager only detects when resource pruning should be considered based on your configured thresholds and observation period. It is your responsibility to implement the actual pruning actions within your application when resource_manager.should_prune() returns True.

* **NVML and GPU Monitoring:** GPU monitoring relies on the NVIDIA Management Library (NVML). Ensure you have NVIDIA drivers and the pynvml Python library installed if you need GPU monitoring. If pynvml is not available, GPU monitoring will be disabled gracefully.

* **Network Usage Metric:** The network usage is measured as bytes received and sent per second. This is an estimate of bandwidth usage and might not directly reflect utilization as a percentage of maximum network capacity, which is harder to determine portably. Adjust the max_network_usage_bytes_per_second threshold according to your application's needs and network environment.

* **close() method:** It's crucial to call resource_manager.close() when you are finished using the ResourceManager, especially in GPU monitoring scenarios, to ensure NVML resources are properly released.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project URL

<https://github.com/saviornt/ResourceManager>
