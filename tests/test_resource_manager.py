import threading
import time
import pytest

from ..src.resource_manager import ResourceManager

def cpu_stress(stop_event):
    """Busy loop that stresses the CPU until stop_event is set."""
    while not stop_event.is_set():
        # Perform a non-optimized calculation to load the CPU
        sum(i * i for i in range(10000))

def allocate_memory():
    """Allocates a large list to stress memory usage.
    Adjust the size to exceed the ResourceManager memory threshold.
    """
    # Here, each Python int takes more than 8 bytes but for testing,
    # we assume this allocation will quickly push memory over a low threshold.
    mem_block = [0] * (10 * 1024 * 1024)  # roughly 10 million ints
    # Hold the memory for a short while to let the stress build up.
    time.sleep(5)
    return mem_block

@pytest.mark.stress
def test_resource_manager_stress():
    """
    Stress test for ResourceManager:
      - Lower the thresholds so that even moderate CPU and memory usage trigger pruning.
      - Start a CPU stress thread and allocate memory.
      - Verify that should_prune() eventually returns True.
    """
    # Lower thresholds for testing purposes:
    # For instance, setting max_memory_usage_mb to 50 MB and max_cpu_usage_percent to 5%.
    rm = ResourceManager(max_memory_usage_mb=50, max_cpu_usage_percent=5, max_gpu_usage_percent=100, check_interval=0.5)
    
    # Start CPU stress in a separate thread.
    stop_event = threading.Event()
    cpu_thread = threading.Thread(target=cpu_stress, args=(stop_event,))
    cpu_thread.start()
    
    # Allocate memory to create stress.
    mem_block = allocate_memory()
    
    # Give a couple of seconds for the stress to build.
    time.sleep(2)
    
    # Check several times whether should_prune() returns True.
    triggered = False
    for _ in range(10):
        if rm.should_prune():
            triggered = True
            break
        time.sleep(0.5)
    
    # Stop the CPU stress thread.
    stop_event.set()
    cpu_thread.join()
    
    # Clean up the allocated memory.
    del mem_block
    
    print("Stress Test Result: should_prune triggered =", triggered)
    # Assert that under stress conditions, the resource manager detected high load.
    assert triggered, "ResourceManager did not trigger prune under stress conditions."
    
    rm.close()
