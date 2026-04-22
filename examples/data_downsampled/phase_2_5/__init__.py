# Initialize the app package
import os
import sys

# Add the plugins directory to the system path to ensure plugins can be dynamically loaded
sys.path.append(os.path.join(os.path.dirname(__file__), 'plugins'))
