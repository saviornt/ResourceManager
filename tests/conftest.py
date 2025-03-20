"""
Configuration file for pytest to set up fixtures and handle common test setup/teardown.
"""
import pytest
import os
import sys
import logging
import atexit
import threading
import signal
import gc
import time
import json
import datetime
from typing import Set, Dict, Any, List
from _pytest.runner import CallInfo
from _pytest.reports import TestReport

# Import our logging utility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging_utils import setup_logging, get_logger, shutdown_logging

# Setup logging for tests
setup_logging(logs_dir="./logs", log_level=logging.DEBUG, app_name="resource_manager_test_conftest")
logger = get_logger("conftest")

# Global tracking of resources for cleanup
_threads_to_cleanup: Set[threading.Thread] = set()  # Changed to set to avoid duplicates
_cleanup_in_progress = False
_thread_register_lock = threading.Lock()

# Test results tracking
_test_results: List[Dict[str, Any]] = []
_test_start_times: Dict[str, float] = {}
_test_results_lock = threading.Lock()

def register_thread_for_cleanup(thread: threading.Thread) -> None:
    """Register a thread for cleanup at the end of testing."""
    if thread is None or not isinstance(thread, threading.Thread):
        return
        
    with _thread_register_lock:
        _threads_to_cleanup.add(thread)  # Use add for set operations
        logger.debug(f"Registered thread {thread.name} for cleanup (total: {len(_threads_to_cleanup)})")

def _force_exit_threads() -> None:
    """Force threads to exit by setting them as daemon threads"""
    with _thread_register_lock:
        persistent_threads = [t for t in _threads_to_cleanup if t.is_alive()]
        if persistent_threads:
            # Use print instead of logging to avoid I/O closed file errors
            print(f"Attempting to force {len(persistent_threads)} persistent threads to exit")
            for thread in persistent_threads:
                try:
                    thread._daemonic = True  # Hack to allow Python to exit anyway
                    print(f"Forced thread {thread.name} to daemon mode")
                except Exception as e:
                    print(f"Failed to force thread {thread.name} to daemon mode: {e}")

def save_test_results() -> None:
    """Save test results to a file in the results directory."""
    try:
        if not _test_results:
            logger.warning("No test results to save")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "./tests/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Group test results by module for better organization
        module_results = {}
        for result in _test_results:
            module_name = result.get("module", "")
            # Extract the short module name (last part after the dot)
            short_module = module_name.split(".")[-1] if module_name else "unknown"
            if short_module not in module_results:
                module_results[short_module] = []
            module_results[short_module].append(result)
        
        # If we have multiple modules, create a file for each module
        if not module_results:
            logger.warning("No module information found in test results")
            filename = f"{results_dir}/test_results_unknown_{timestamp}.json"
            results_data = {
                "timestamp": timestamp,
                "total_tests": len(_test_results),
                "passed": sum(1 for r in _test_results if r["outcome"] == "passed"),
                "failed": sum(1 for r in _test_results if r["outcome"] == "failed"),
                "skipped": sum(1 for r in _test_results if r["outcome"] == "skipped"),
                "total_duration": sum(r["duration"] for r in _test_results),
                "test_results": _test_results
            }
            with open(filename, "w") as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Test results saved to {filename}")
        else:
            saved_files = []
            for module_name, results in module_results.items():
                filename = f"{results_dir}/{module_name}_test_results_{timestamp}.json"
                results_data = {
                    "timestamp": timestamp,
                    "module": module_name,
                    "total_tests": len(results),
                    "passed": sum(1 for r in results if r["outcome"] == "passed"),
                    "failed": sum(1 for r in results if r["outcome"] == "failed"),
                    "skipped": sum(1 for r in results if r["outcome"] == "skipped"),
                    "total_duration": sum(r["duration"] for r in results),
                    "test_results": results
                }
                with open(filename, "w") as f:
                    json.dump(results_data, f, indent=2)
                saved_files.append(filename)
            
            logger.info(f"Test results saved to {', '.join(saved_files)}")
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before any tests run."""
    logger.info("Setting up test environment")
    
    # Make sure results directory exists
    os.makedirs("./tests/results", exist_ok=True)
    
    # Register signal handlers
    def handle_signal(sig, frame):
        print(f"Received signal {sig} in conftest, cleaning up...")
        # Save test results before exit
        save_test_results()
        # Shutdown logging before cleanup
        shutdown_logging()
        cleanup_threads(force=True, exit_after=True)
        # Force immediate exit on signals
        os._exit(1)
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    yield
    
    # Save test results
    save_test_results()
    
    # Cleanup all registered threads
    logger.info("Cleaning up test environment")
    # First shutdown logging to prevent I/O errors
    shutdown_logging()
    # Then clean up threads
    cleanup_threads(force=True)
    
    # Force garbage collection
    gc.collect()
    
    print("Test environment cleanup complete")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Record test start time."""
    test_id = item.nodeid
    with _test_results_lock:
        _test_start_times[test_id] = time.time()

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Clean up any test-specific resources."""
    pass  # Any test-specific cleanup can go here

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Process test results and record metrics."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        test_id = item.nodeid
        
        # Get start time and calculate duration
        start_time = _test_start_times.get(test_id, time.time())
        duration = time.time() - start_time
        
        # Extract test parameters if any
        params = {}
        if hasattr(item, "callspec"):
            params = item.callspec.params
        
        # Create result entry
        test_result = {
            "test_id": test_id,
            "name": item.name,
            "module": item.module.__name__ if hasattr(item, "module") else "",
            "outcome": report.outcome,
            "duration": duration,
            "parameters": params,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Add any error information
        if report.outcome == "failed":
            test_result["error"] = str(report.longrepr) if hasattr(report, "longrepr") else "Unknown error"
        
        # Add to results list
        with _test_results_lock:
            _test_results.append(test_result)
            # Clean up start time
            if test_id in _test_start_times:
                del _test_start_times[test_id]

def cleanup_threads(force=False, exit_after=False) -> None:
    """Clean up any remaining threads."""
    global _cleanup_in_progress
    if _cleanup_in_progress and not force:
        return
    
    _cleanup_in_progress = True
    
    try:
        active_threads = [t for t in _threads_to_cleanup if t and t.is_alive()]
        if active_threads:
            print(f"Found {len(active_threads)} active threads during cleanup")
            for thread in active_threads:
                print(f"Attempting to join thread {thread.name}")
                try:
                    thread.join(timeout=1.0)  # Shorter initial timeout
                    if thread.is_alive():
                        print(f"Thread {thread.name} did not terminate within initial timeout")
                        # If force is True, we're shutting down so we can try a final join
                        if force:
                            # Try one more time with a shorter timeout
                            thread.join(timeout=0.5)
                    else:
                        print(f"Thread {thread.name} terminated successfully")
                except Exception as e:
                    print(f"Error joining thread {thread.name}: {e}")
                    
        # Check if we still have running threads
        still_active = [t for t in _threads_to_cleanup if t and t.is_alive()]
        if still_active and force:
            print(f"Still have {len(still_active)} active threads after join attempts")
            _force_exit_threads()
    except Exception as e:
        print(f"Error during thread cleanup: {e}")
    finally:
        _threads_to_cleanup.clear()
        _cleanup_in_progress = False
        print("Thread cleanup complete")
        
        if exit_after:
            print("Forced exit requested after cleanup")

# Register thread cleanup function to run at exit
@atexit.register
def cleanup_at_exit():
    """Cleanup function that runs at process exit"""
    try:
        print("Final cleanup at exit")
        # Save test results before shutdown
        save_test_results()
        # First shut down logging
        shutdown_logging()
        # Then clean up threads
        cleanup_threads(force=True)
    except Exception:
        # Ignore any errors during shutdown
        pass

# Override Thread.start to automatically register threads
original_thread_start = threading.Thread.start
def patched_thread_start(self, *args, **kwargs):
    """Patch Thread.start to auto-register threads for cleanup"""
    result = original_thread_start(self, *args, **kwargs)
    register_thread_for_cleanup(self)
    return result

threading.Thread.start = patched_thread_start 