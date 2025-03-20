# This file is necessary for pytest to recognize the directory as a package
# It can remain empty but it's useful to add test configuration here if needed

import os
import sys

# Add the parent directory to the path to make module imports work correctly in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
