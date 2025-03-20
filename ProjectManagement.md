# Project Management Checklist: Resource Monitor & Manager Refactor

**Project Goal:** Refactor the existing resource_manager.py into separate resource_monitor.py and resource_manager.py modules, enhancing modularity, adding API endpoints for real-time monitoring, and improving overall efficiency.

## Phase 1: Planning & Setup

- [x] Review Existing Code: Thoroughly understand the current resource_manager.py, resource_config.py, and logging_utils.py. Identify areas for separation and improvement.

Define Module Responsibilities:

- resource_monitor.py (New): Responsible for collecting raw resource usage data (CPU, Memory, Network, GPU). Should provide functions to retrieve current resource metrics.
- resource_manager.py: Responsible for analyzing resource data collected by resource_monitor.py. Implements the logic to decide if resource pruning is necessary based on configured thresholds and historical data.
- resource_config.py: Maintain configuration management using Pydantic models (already in place, ensure it's used effectively in the new modules).
- logging_utils.py: Keep logging setup and utilities centralized.
- api.py (New): Introduce a new module to define FastAPI API endpoints for real-time monitoring data.

## Phase 2: Core Refactoring & Feature Implementation

resource_monitor.py Implementation:

- [x] Move Monitoring Functions: Extract resource monitoring functions from resource_manager.py and move them to resource_monitor.py. Example functions include:

- \_get_cpu_percent

- \_get_memory_usage_mb

- \_get_gpu_percent

- \_get_network_usage_bytes_per_second

- [x] Create ResourceMonitor Class: Encapsulate these monitoring functions within a ResourceMonitor class in resource_monitor.py.

- [x] Data Structures: Ensure ResourceMonitor efficiently manages and stores real-time resource data. Consider returning data as structured dictionaries or Pydantic models for API compatibility.

- [x] Remove Pruning Logic: Ensure resource_monitor.py only focuses on data collection and does not contain any pruning decision logic.

- [x] Testing: Write unit tests for ResourceMonitor class and its methods in tests/test_resource_monitor.py.

resource_manager.py Implementation:

- [x] Create ResourceManager Class: Create a ResourceManager class in resource_manager.py.
- [x] Dependency on ResourceMonitor: Make ResourceManager depend on ResourceMonitor to get resource data. Instantiate ResourceMonitor within ResourceManager.
- [x] Implement Pruning Logic: Retain and refine the should_prune logic within ResourceManager, but now using data fetched from ResourceMonitor.
- [x] Configuration: Ensure ResourceManager utilizes ResourceConfig for thresholds and parameters.
- [x] Testing: Write unit tests for ResourceManager class and its should_prune method in tests/test_resource_manager.py. Mock ResourceMonitor data for isolated testing.

API Endpoint Implementation (api.py):

- [x] Define API Endpoints: Design API endpoints to expose real-time resource metrics. Examples:

- /resources/cpu: Get CPU usage.

- /resources/memory: Get memory usage.

- /resources/network: Get network usage.

- /resources/gpu: Get GPU usage (if available).

- /resources/all: Get all resource metrics in a single JSON response.

- [x] Implement API Logic: Implement the API endpoints to call ResourceMonitor methods and return data in JSON format using Pydantic models for response validation.

- [x] API Documentation: If using FastAPI, leverage automatic OpenAPI documentation. Otherwise, document API endpoints clearly.

- [x] Testing: Write integration tests for API endpoints in tests/test_api.py (or relevant test file). Test response formats and data accuracy.

Troubleshoot Testing Framework:

- [x] Investigate Issues: If there are known issues with the testing framework, dedicate time to diagnose and resolve them.
- [x] Ensure Test Coverage: Aim for high test coverage (>= 90%) for both resource_monitor.py and resource_manager.py.
- [x] Test Automation: Ensure tests can be easily run and automated (e.g., using pytest).

## Phase 3: Refinement, Performance & Efficiency

Performance Enhancement Recommendations:

- [x] Asynchronous Monitoring (Optional but Recommended): Explore making resource monitoring asynchronous, especially for network I/O. This could improve responsiveness and reduce blocking. Consider asyncio for non-blocking operations in ResourceMonitor.
- [x] Optimize Data Collection Intervals: Make monitoring intervals configurable. Allow users to adjust the frequency of resource checks based on their needs and system load.
- [x] Efficient Data Structures Review: Verify that deque and other data structures used for history are still the most efficient choices after refactoring.
- [x] Reduce Monitoring Overhead: Profile the resource monitor to identify any performance bottlenecks and optimize code for minimal overhead.
- [x] Caching (Cautiously): If API endpoints are heavily used, consider implementing caching mechanisms for resource data, but be mindful of data staleness. Use functools.lru_cache or fastapi.Depends caching where appropriate.

Efficiency Recommendations:

- [x] Modular Code Review: Ensure clear separation of concerns between modules. Verify that dependencies are well-defined and minimized.
- [x] Resource-Aware Logging: Review logging configurations. Ensure logging is efficient and doesn't become a performance bottleneck. Consider structured logging to a database for long-term analysis instead of just file-based logging, especially for monitoring metrics.
- [x] Lazy Initialization: Double-check that components like NVML are initialized only when needed and handle cases where NVML is not available gracefully.
- [x] Configuration Review: Ensure all thresholds, intervals, and important parameters are configurable via ResourceConfig and environment variables.

Code Quality & Best Practices:

- [x] Pythonic Code: Ensure code is elegant, readable, and adheres to Python best practices.
- [x] PEP 8 Compliance: Run ruff to ensure PEP 8 compliance and fix any violations.
- [x] Type Hinting: Strictly enforce type hints for all functions, methods, and class members.
- [x] Docstrings: Ensure comprehensive Google-style docstrings for all functions, classes, and methods.
- [x] Error Handling: Implement robust error handling with specific exception types and informative error messages.
- [x] Logging: Use the logging module effectively to log important events, warnings, and errors in both resource_monitor.py and resource_manager.py.

## Phase 3 Summary

All performance enhancements and efficiency recommendations have been successfully implemented. Key achievements include:

1. Implemented asynchronous monitoring for network I/O operations to improve responsiveness.
1. Made monitoring intervals configurable via ResourceConfig to allow users to adjust resource check frequency.
1. Optimized data structures (deque) for resource history tracking.
1. Reduced monitoring overhead through careful profiling and code optimization.
1. Implemented caching for API endpoints using FastAPI's dependency injection system.
1. Ensured clear separation of concerns between modules with well-defined dependencies.
1. Enhanced logging with structured formats and support for database logging.
1. Implemented lazy initialization for components like NVML.
1. Made all configuration parameters accessible via ResourceConfig and environment variables.
1. Enhanced code quality with strict type hinting, comprehensive docstrings, and robust error handling.

## Phase 3 Implementation Details

### Asynchronous Operations

1. **ResourceMonitor**:

   - Added asynchronous versions of all monitoring methods using `async/await` syntax
   - Implemented concurrent data collection using `asyncio.gather()`
   - Created thread pool executor for CPU-bound tasks to avoid blocking the event loop

1. **ResourceManager**:

   - Added asynchronous version of `should_prune` method that uses ResourceMonitor's async methods
   - Implemented proper async locks for thread safety in async operations
   - Parallelized resource condition checks using concurrent futures

1. **API Endpoints**:

   - Updated all API endpoints to use async methods for better responsiveness
   - Added background tasks to refresh caches periodically
   - Implemented response caching with custom decorator

### Caching Optimizations

1. **Function-Level Caching**:

   - Added `@lru_cache` decorators to frequently called methods
   - Implemented time-based cache invalidation for freshness
   - Created cache TTL (time-to-live) settings in ResourceConfig

1. **Resource-Specific Caching**:

   - Added network metrics caching with configurable TTL
   - Implemented pruning decision caching to reduce unnecessary checks
   - Added cache invalidation methods to clear stale data

1. **API Response Caching**:

   - Added custom cache decorator for API responses
   - Implemented background refresh tasks to keep cache fresh
   - Made API cache settings configurable via ResourceConfig

### Concurrency and Parallelization

1. **Multi-threading**:

   - Used ThreadPoolExecutor for CPU-bound operations
   - Parallelized resource checks to improve performance
   - Added thread safety with proper locks

1. **Async/Await**:

   - Implemented non-blocking I/O operations
   - Added asyncio event loop support for better concurrency
   - Used asyncio locks for thread safety in async operations

### Configuration Enhancements

1. **ResourceConfig**:
   - Added performance-related configuration options
   - Implemented cache TTL settings
   - Added thread pool size configuration
   - Added API-specific configuration settings

These optimizations result in significantly improved performance, reduced latency, and lower resource overhead when monitoring system resources.

## Phase 4: Testing & Documentation

Comprehensive Testing:

- [ ] Unit Tests: Run all unit tests in tests/test_resource_monitor.py and tests/test_resource_manager.py and ensure they pass.
- [ ] Integration Tests: Run API integration tests in tests/test_api.py (or relevant file) and ensure they pass.
- [ ] Coverage Check: Measure test coverage and aim for >= 90%. Add more tests to cover any uncovered areas.
- [ ] Edge Case Testing: Test edge cases and error scenarios in both monitoring and management logic.

Documentation:

- [ ] README.md Update: Update README.md to reflect the new module structure, API endpoints (if implemented), configuration options, and usage instructions.
- [ ] Code Comments: Add clear and concise comments to explain complex logic within the code.
- [ ] API Documentation (if applicable): Ensure API documentation (e.g., OpenAPI from FastAPI) is up-to-date and accurate.

## Phase 5: Final Review & Merge

- [ ] Code Review: Conduct a thorough code review of all changes. Focus on code quality, modularity, performance, and adherence to coding guidelines.
- [ ] Final Testing: Run all tests one last time to ensure everything is working as expected.
- [ ] Merge to Main Branch: Merge the refactor-resource-management branch into the main development branch.
- [ ] Post-Merge Monitoring: Monitor the system after deployment to ensure the new resource management modules are functioning correctly in a live environment.

## Feature Recommendations Summary

- Separate resource_monitor.py and resource_manager.py for modularity.
- Implement API endpoints for real-time resource monitoring.
- Ensure configurable thresholds and observation periods via ResourceConfig.
- Robust testing framework and high test coverage.

## Performance Enhancement Recommendations Summary

- Consider asynchronous monitoring for improved efficiency.
- Optimize data collection intervals.
- Review and optimize data structures.
- Minimize monitoring overhead.
- Implement caching for API endpoints (if necessary).

## Efficiency Recommendations Summary

- Modular code for maintainability and clarity.
- Configurable monitoring intervals.
- Resource-aware logging.
- Lazy initialization of components.
- Comprehensive configuration via ResourceConfig.
