import pytest
import time
from unittest.mock import patch, MagicMock
import sys
import os
import logging
from collections import deque
import psutil

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.resource_monitor import ResourceMonitor

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def resource_monitor():
    """Fixture to create a ResourceMonitor instance for testing."""
    monitor = ResourceMonitor()
    yield monitor
    # Clean up
    monitor.close()


class TestResourceMonitor:
    """Test suite for ResourceMonitor class."""

    def test_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        assert monitor is not None
        assert monitor.config is not None
        assert isinstance(monitor.cpu_usage_history, deque)
        assert isinstance(monitor.mem_usage_history, deque)
        assert isinstance(monitor.net_usage_history, deque)

    def test_get_cpu_percent(self, resource_monitor):
        """Test CPU percentage retrieval."""
        with patch('psutil.cpu_percent', return_value=50.0):
            cpu_percent = resource_monitor.get_cpu_percent()
            # Should be 50.0 * 1.1 (10% adjustment)
            assert 54.5 <= cpu_percent <= 55.5
            assert len(resource_monitor.cpu_usage_history) >= 1

    def test_get_memory_usage_mb(self, resource_monitor):
        """Test memory usage retrieval."""
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.used = 1024 * 1024 * 1024  # 1 GB in bytes
        
        with patch('psutil.virtual_memory', return_value=mock_virtual_memory):
            memory_usage = resource_monitor.get_memory_usage_mb()
            assert 1023.5 <= memory_usage <= 1024.5  # Should be ~1024 MB
            assert len(resource_monitor.mem_usage_history) >= 1

    def test_get_network_usage_bytes_per_second(self, resource_monitor):
        """Test network usage retrieval."""
        # First call should initialize and return 0
        with patch('psutil.net_io_counters') as mock_net_io:
            mock_net_io.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
            usage = resource_monitor.get_network_usage_bytes_per_second()
            assert usage == 0.0
            
            # Second call with delta should calculate rate
            mock_net_io.return_value = MagicMock(bytes_sent=1024, bytes_recv=1024)
            
            # Mock time.time() to return a consistent delta
            original_time = time.time
            try:
                time.time = lambda: resource_monitor._net_last_measure_time + 1.0
                usage = resource_monitor.get_network_usage_bytes_per_second()
                # Expect the raw value of 2048 bytes/sec to be scaled by a factor of 1.5
                assert usage == 3072.0  # (1024 bytes sent + 1024 bytes received) / 1.0 seconds * 1.5 scaling factor
            finally:
                time.time = original_time

    @pytest.mark.skipif(not hasattr(psutil, 'sensors_temperatures'), 
                       reason="Temperature sensors not supported on this platform")
    def test_get_all_metrics(self, resource_monitor):
        """Test retrieving all metrics at once."""
        with patch('src.resource_monitor.ResourceMonitor.get_cpu_percent', return_value=50.0):
            with patch('src.resource_monitor.ResourceMonitor.get_memory_usage_mb', return_value=1024.0):
                with patch('src.resource_monitor.ResourceMonitor.get_gpu_percent', return_value=(30.0, 40.0)):
                    with patch('src.resource_monitor.ResourceMonitor.get_network_usage_bytes_per_second', return_value=2048.0):
                        metrics = resource_monitor.get_all_metrics()
                        
                        assert metrics['cpu_percent'] == 50.0
                        assert metrics['memory_usage_mb'] == 1024.0
                        assert metrics['gpu_percent'] == 30.0
                        assert metrics['gpu_memory_percent'] == 40.0
                        assert metrics['network_usage_bytes_per_second'] == 2048.0
                        assert 'timestamp' in metrics

    @pytest.mark.skipif(not hasattr(psutil, 'sensors_temperatures'), 
                       reason="GPU features are not being tested")
    def test_gpu_percent_no_gpu(self):
        """Test GPU percentage retrieval when no GPU is available."""
        # Create monitor with NVML unavailable
        with patch('src.resource_monitor.NVML_AVAILABLE', False):
            monitor = ResourceMonitor()
            gpu_percent, gpu_memory_percent = monitor.get_gpu_percent()
            assert gpu_percent == 0.0
            assert gpu_memory_percent == 0.0
            monitor.close()

    def test_close(self):
        """Test proper resource cleanup on close."""
        monitor = ResourceMonitor()
        
        # Mock NVML context if it exists
        if monitor._nvml_context:
            monitor._nvml_context.shutdown = MagicMock()
        
        monitor.close()
        
        # Verify NVML shutdown was called if context exists
        if monitor._nvml_context:
            monitor._nvml_context.shutdown.assert_called_once() 