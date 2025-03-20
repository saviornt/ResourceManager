#!/usr/bin/env python3
import os
import sys
import pytest
import signal
import logging
import threading
import gc

# Add the parent directory to sys.path to make src module imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging_utils import setup_logging, shutdown_logging

# Global flag to track if we're in cleanup mode
_cleanup_in_progress = False

# Store all threads that need to be cleaned up
_active_threads = set()
_cleanup_lock = threading.Lock()

def register_cleanup_thread(thread):
    """Register a thread for cleanup at exit"""
    with _cleanup_lock:
        _active_threads.add(thread)

def _force_shutdown_threads():
    """Force terminate threads that don't respond to normal shutdown"""
    global _active_threads
    
    with _cleanup_lock:
        stubborn_threads = [t for t in _active_threads if t.is_alive()]
        
        if not stubborn_threads:
            return
        
        # Set threads as daemon to let the interpreter exit anyway
        for thread in stubborn_threads:
            try:
                # Convert to daemon thread to allow interpreter to exit
                # This is a bit of a hack but can help with exit
                thread._daemonic = True
                # Use print instead of logging here to avoid closed file errors
                print(f"Forced thread {thread.name} to daemon mode")
            except Exception:
                pass

def cleanup_all_threads(force=False):
    """Clean up all registered threads"""
    global _cleanup_in_progress
    with _cleanup_lock:
        if _cleanup_in_progress and not force:
            return
        
        _cleanup_in_progress = True
        
        try:
            # Use print instead of logging for cleanup messages
            thread_count = len(_active_threads)
            print(f"Cleaning up {thread_count} threads at exit")
            
            # First attempt: join normally
            for thread in list(_active_threads):
                if thread and thread.is_alive():
                    try:
                        print(f"Joining thread {thread.name}")
                        thread.join(timeout=1.0)  # Reduced timeout
                    except Exception as e:
                        # Print instead of log to avoid errors
                        print(f"Error joining thread {thread.name}: {e}")
            
            # Second attempt: check which threads are still alive
            still_alive = [t for t in _active_threads if t and t.is_alive()]
            if still_alive:
                print(f"{len(still_alive)} threads still alive after first join attempt")
                
                # Try one more join with shorter timeout
                for thread in still_alive:
                    try:
                        thread.join(timeout=0.5)
                    except Exception:
                        pass
                
                # For any thread still alive, try to force terminate
                if force:
                    _force_shutdown_threads()
            
            _active_threads.clear()
        except Exception as e:
            # Handle any exceptions during cleanup
            print(f"Error during thread cleanup: {e}")
            pass
        
        print("Thread cleanup complete")
        
        _cleanup_in_progress = False

# Monkey patch Thread.start to automatically register threads
original_start = threading.Thread.start
def patched_start(self, *args, **kwargs):
    result = original_start(self, *args, **kwargs)
    register_cleanup_thread(self)
    return result
threading.Thread.start = patched_start

def run_tests():
    """Run the tests with the proper setup and teardown."""
    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Make sure results directory exists
    os.makedirs("tests/results", exist_ok=True)
    
    # Setup logging
    setup_logging(logs_dir="./logs", log_level=logging.DEBUG, app_name="resource_manager_test")
    
    # Log the start of tests
    logging.info("Starting ResourceManager tests")
    
    # Run the tests
    args = [
        "--verbose",
        "-xvs",  # Show extra test info and stop on first error
        "--capture=sys",  # Capture stdout/stderr
        "tests/",
    ]
    
    # Add any command-line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    try:
        # Run pytest
        result = pytest.main(args)
        
        # Log the test results before shutting down logging
        if result == 0:
            logging.info("All tests passed successfully")
        else:
            logging.error(f"Tests failed with exit code {result}")
        
        # Make sure we disable logging properly before any cleanup
        shutdown_logging()
        
        # Run explicit cleanup on threads
        cleanup_all_threads()
        
        return result
    except Exception as e:
        print(f"Error during test execution: {e}")
        # Disable logging immediately on error
        shutdown_logging()
        return 1

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    def signal_handler(sig, frame):
        # Disable logging before cleanup to avoid potential errors
        print(f"Received signal {sig}, exiting gracefully...")
        # Disable logging before any cleanup
        shutdown_logging()
        # Clean up threads
        cleanup_all_threads(force=True)
        # Force immediate exit
        os._exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the tests
    try:
        # Run tests
        exit_code = run_tests()
        
        # Print instead of log to avoid I/O errors
        print("Tests completed, performing final cleanup")
        
        # Make sure logging is disabled before final cleanup
        shutdown_logging()
        
        # Final explicit cleanup
        cleanup_all_threads(force=True)  
        
        # Force GC to help release resources
        gc.collect()
        
        # Exit immediately
        os._exit(exit_code)  # Use os._exit to force immediate exit
    except KeyboardInterrupt:
        print("Test execution interrupted by user (KeyboardInterrupt)")
        shutdown_logging()
        cleanup_all_threads(force=True)
        os._exit(1)
    except Exception as e:
        print(f"Unexpected error during test execution: {e}")
        shutdown_logging()
        cleanup_all_threads(force=True)
        os._exit(1) 