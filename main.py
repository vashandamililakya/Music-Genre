"""
main.py
=======
Entry point for the AI Music Genre Classifier API server.

Run
---
    python main.py                    # development server (port 5000)
    python main.py --port 8080        # custom port
    python main.py --debug            # debug mode with auto-reload

Production
----------
Use a proper WSGI server instead of Flask's dev server:

    # gunicorn (recommended)
    pip install gunicorn
    gunicorn "main:application" --workers 4 --bind 0.0.0.0:5000

    # waitress (Windows-friendly)
    pip install waitress
    waitress-serve --host=0.0.0.0 --port=5000 main:application
"""

import argparse
import logging

from app import create_app
from config import Config

logger = logging.getLogger(__name__)

# WSGI-compatible application object (used by gunicorn / waitress)
application = create_app(Config)


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
