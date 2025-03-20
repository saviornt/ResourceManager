import pytest
import sys
import os
import logging
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app, get_resource_monitor
from src.resource_monitor import ResourceMonitor

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

# Create a test client
client = TestClient(app)


# Override the dependency
@pytest.fixture(autouse=True)
def mock_get_resource_monitor():
    """
    Mock the ResourceMonitor dependency for all tests.
    This ensures we don't actually use system resources during tests.
    """
    mock_monitor = MagicMock(spec=ResourceMonitor)
    
    # Set up return values for synchronous methods
    mock_monitor.get_cpu_percent.return_value = 50.0
    mock_monitor.get_memory_usage_mb.return_value = 1024.0
    mock_monitor.get_gpu_percent.return_value = (30.0, 40.0)
    mock_monitor.get_network_usage_bytes_per_second.return_value = 2048.0
    mock_monitor.get_all_metrics.return_value = {
        "timestamp": 1620000000.0,
        "cpu_percent": 50.0,
        "memory_usage_mb": 1024.0,
        "gpu_percent": 30.0,
        "gpu_memory_percent": 40.0,
        "network_usage_bytes_per_second": 2048.0
    }
    
    # Set up return values for asynchronous methods
    mock_monitor.get_cpu_percent_async = AsyncMock(return_value=50.0)
    mock_monitor.get_memory_usage_mb_async = AsyncMock(return_value=1024.0)
    mock_monitor.get_gpu_percent_async = AsyncMock(return_value=(30.0, 40.0))
    mock_monitor.get_network_usage_bytes_per_second_async = AsyncMock(return_value=2048.0)
    mock_monitor.get_all_metrics_async = AsyncMock(return_value={
        "timestamp": 1620000000.0,
        "cpu_percent": 50.0,
        "memory_usage_mb": 1024.0,
        "gpu_percent": 30.0,
        "gpu_memory_percent": 40.0,
        "network_usage_bytes_per_second": 2048.0
    })
    
    # Set properties
    mock_monitor._cpu_last_measure_time = 1620000000.0
    mock_monitor._net_last_measure_time = 1620000000.0
    
    # Replace the dependency
    original_get_resource_monitor = app.dependency_overrides.get(get_resource_monitor, None)
    app.dependency_overrides[get_resource_monitor] = lambda: mock_monitor
    
    yield mock_monitor
    
    # Clean up
    if original_get_resource_monitor:
        app.dependency_overrides[get_resource_monitor] = original_get_resource_monitor
    else:
        del app.dependency_overrides[get_resource_monitor]


class TestAPI:
    """Test suite for the resource monitoring API endpoints."""
    
    def test_get_cpu_usage(self, mock_get_resource_monitor):
        """Test the CPU usage endpoint."""
        response = client.get("/resources/cpu")
        assert response.status_code == 200
        data = response.json()
        assert data["cpu_percent"] == 50.0
        assert data["timestamp"] == 1620000000.0
        # Check that the async method was called
        mock_get_resource_monitor.get_cpu_percent_async.assert_called_once()
    
    def test_get_memory_usage(self, mock_get_resource_monitor):
        """Test the memory usage endpoint."""
        response = client.get("/resources/memory")
        assert response.status_code == 200
        data = response.json()
        assert data["memory_usage_mb"] == 1024.0
        assert data["timestamp"] == 1620000000.0
        # Check that the async method was called
        mock_get_resource_monitor.get_memory_usage_mb_async.assert_called_once()
    
    def test_get_gpu_usage(self, mock_get_resource_monitor):
        """Test the GPU usage endpoint."""
        response = client.get("/resources/gpu")
        assert response.status_code == 200
        data = response.json()
        assert data["gpu_percent"] == 30.0
        assert data["gpu_memory_percent"] == 40.0
        assert data["timestamp"] == 1620000000.0
        # Check that the async method was called
        mock_get_resource_monitor.get_gpu_percent_async.assert_called_once()
    
    def test_get_network_usage(self, mock_get_resource_monitor):
        """Test the network usage endpoint."""
        response = client.get("/resources/network")
        assert response.status_code == 200
        data = response.json()
        assert data["network_usage_bytes_per_second"] == 2048.0
        assert data["timestamp"] == 1620000000.0
        # Check that the async method was called
        mock_get_resource_monitor.get_network_usage_bytes_per_second_async.assert_called_once()
    
    def test_get_all_resources(self, mock_get_resource_monitor):
        """Test the combined resources endpoint."""
        response = client.get("/resources/all")
        assert response.status_code == 200
        data = response.json()
        assert data["cpu_percent"] == 50.0
        assert data["memory_usage_mb"] == 1024.0
        assert data["gpu_percent"] == 30.0
        assert data["gpu_memory_percent"] == 40.0
        assert data["network_usage_bytes_per_second"] == 2048.0
        assert data["timestamp"] == 1620000000.0
        # Check that the async method was called
        mock_get_resource_monitor.get_all_metrics_async.assert_called_once() 