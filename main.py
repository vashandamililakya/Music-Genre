"""
main.py
=======
Entry point for the AI Music Genre Classifier API server.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path (fixes Render/gunicorn module resolution)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

from app import create_app
from config import Config

logger = logging.getLogger(__name__)

# WSGI-compatible application object (used by gunicorn / waitress)
application = create_app(Config)
app = application  # alias for flask CLI compatibility


def main():
    parser = argparse.ArgumentParser(description="Music Genre Classifier API server")
    parser.add_argument("--host",  default=Config.HOST,  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  default=Config.PORT,  type=int, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true",  help="Enable Flask debug mode")
    args = parser.parse_args()

    logger.info("Starting server on http://%s:%d", args.host, args.port)
    application.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
