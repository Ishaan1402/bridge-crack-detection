import os
import sys

# Setting project root directory into Python path for tests to locate src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
