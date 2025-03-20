from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
import json
import logging
import os
import sys
import pytest
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.endpoints import app, get_resource_monitor, get_cpu_usage, get_memory_usage, get_gpu_usage, get_network_usage, get_all_resources
from src.resource_monitor import ResourceMonitor

# Configure logging for tests
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create results directory if it doesn't exist
results_dir = Path("./tests/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Create a test client
client = TestClient(app)

# Override the dependency
@pytest.fixture
def mock_get_resource_monitor():
    """Create a mock ResourceMonitor for testing."""
    mock = MagicMock(spec=ResourceMonitor)
    
    # Mock CPU usage
    mock.get_cpu_percent_async = AsyncMock(return_value=50.0)
    mock._cpu_last_measure_time = 1620000000.0
    
    # Mock memory usage
    mock.get_memory_usage_mb_async = AsyncMock(return_value=1024.0)
    
    # Mock GPU usage
    mock.get_gpu_percent_async = AsyncMock(return_value=(30.0, 40.0))
    
    # Mock network usage
    mock.get_network_usage_bytes_per_second_async = AsyncMock(return_value=2048.0)
    mock._net_last_measure_time = 1620000000.0
    
    # Mock all metrics
    mock.get_all_metrics_async = AsyncMock(return_value={
        "cpu_percent": 50.0,
        "memory_usage_mb": 1024.0,
        "gpu_percent": 30.0,
        "gpu_memory_percent": 40.0,
        "network_usage_bytes_per_second": 2048.0,
        "timestamp": 1620000000.0
    })
    
    # Mock the dependency injection
    app.dependency_overrides[get_resource_monitor] = lambda: mock
    yield mock
    
    # Clean up (remove the override)
    app.dependency_overrides.clear()

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
    
    def test_api_error_handling(self):
        """
        Test error handling in API endpoints.
        
        This test checks that all API endpoints properly handle exceptions
        and return appropriate error responses (500 status code).
        """
        # Test CPU endpoint error handling
        # First invalidate the cache for this endpoint
        get_cpu_usage.invalidate_cache()
        with patch('src.resource_monitor.ResourceMonitor.get_cpu_percent_async', 
                   side_effect=Exception("CPU error")):
            response = client.get("/resources/cpu")
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "CPU error" in response.json()["detail"]
        
        # Test memory endpoint error handling
        get_memory_usage.invalidate_cache()
        with patch('src.resource_monitor.ResourceMonitor.get_memory_usage_mb_async', 
                   side_effect=Exception("Memory error")):
            response = client.get("/resources/memory")
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "Memory error" in response.json()["detail"]
        
        # Test GPU endpoint error handling
        get_gpu_usage.invalidate_cache()
        with patch('src.resource_monitor.ResourceMonitor.get_gpu_percent_async', 
                   side_effect=Exception("GPU error")):
            response = client.get("/resources/gpu")
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "GPU error" in response.json()["detail"]
        
        # Test network endpoint error handling
        get_network_usage.invalidate_cache()
        with patch('src.resource_monitor.ResourceMonitor.get_network_usage_bytes_per_second_async', 
                   side_effect=Exception("Network error")):
            response = client.get("/resources/network")
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "Network error" in response.json()["detail"]
        
        # Test all resources endpoint error handling
        get_all_resources.invalidate_cache()
        with patch('src.resource_monitor.ResourceMonitor.get_all_metrics_async', 
                   side_effect=Exception("All resources error")):
            response = client.get("/resources/all")
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "All resources error" in response.json()["detail"]

@pytest.fixture(autouse=True)
def save_test_results(request):
    """Save test results after each test."""
    yield
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"test_api_test_results_{timestamp}.json"
    
    # Get test results (using pytest's report mechanism)
    outcomes = request.node.stash.get("rep_call", None)
    if outcomes:
        test_result = {
            "test_name": request.node.name,
            "timestamp": timestamp,
            "result": "passed" if outcomes.passed else "failed",
            "errors": str(outcomes.longrepr) if not outcomes.passed else None
        }
    else:
        # Fallback in case we can't get report details
        test_result = {
            "test_name": request.node.name,
            "timestamp": timestamp,
            "result": "unknown",
            "errors": None
        }
    
    # Save to file
    with open(result_file, 'w') as f:
        json.dump(test_result, f, indent=4)
        
    logger.info(f"Test results saved to {result_file}")

# Add hooks to capture test outcomes
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # Store test outcomes in the item's stash
    if rep.when == "call":
        item.stash.setdefault("rep_call", rep)
    elif rep.when == "setup":
        item.stash.setdefault("rep_setup", rep)
    elif rep.when == "teardown":
        item.stash.setdefault("rep_teardown", rep) 