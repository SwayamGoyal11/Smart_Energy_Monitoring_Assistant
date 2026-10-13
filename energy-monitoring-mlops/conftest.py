# conftest.py - makes src/ importable from tests/
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
