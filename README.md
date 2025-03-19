# Resource Manager

The Resource Manager is a Python package designed to monitor and manage system resources such as CPU, memory, and GPU usage. It utilizes the Pydantic library for configuration management and the `psutil` library for resource monitoring. If available, it also integrates with NVIDIA's NVML for GPU monitoring.

## Features

- Monitors CPU, memory, and GPU usage.
- Configurable thresholds for resource usage.
- Dynamic resource management based on usage patterns.
- Easy integration with existing Python applications.

## Installation

To install the required dependencies, you can use pip:

```bash
pip install pydantic python-dotenv psutil pynvml
```

## Configuration

The Resource Manager uses environment variables for configuration. You can set the following variables in a `.env` file:

```plaintext
OBSERVATION_PERIOD=10.0
CHECK_INTERVAL=1.0
ROLLING_WINDOW_SIZE=20
TARGET_UTILIZATION=0.9
ENABLED=True
MAX_MEMORY_USAGE_MB=8192.0
MAX_CPU_USAGE_PERCENT=90.0
MAX_GPU_USAGE_PERCENT=90.0
```

## Usage

Here is a basic example of how to use the Resource Manager:

```python
from resource_manager import ResourceManager
from resource_config import ResourceConfig

# Load configuration
config = ResourceConfig()

# Initialize Resource Manager
resource_manager = ResourceManager(config=config)

# Periodically check if pruning is needed
while True:
    if resource_manager.should_prune():
        print("Resource usage exceeded limits. Pruning resources...")
        # Implement your resource pruning logic here
    time.sleep(config.check_interval)

# Clean up resources on exit
resource_manager.close()
```

## Logging

The Resource Manager uses Python's built-in logging module. You can configure the logging level and format as needed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments

- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [psutil](https://psutil.readthedocs.io/en/latest/)
- [NVIDIA NVML](https://developer.nvidia.com/nvidia-management-library-nvml)
