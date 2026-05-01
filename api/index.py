"""
Vercel serverless entry point for Lagi Weather Guardian.
Imports the FastAPI app from the deployment module.
"""

import sys
from pathlib import Path

# Add project root to path so all imports resolve
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / "scripts"))

from deployment.api import app
