#!/usr/bin/env python
import os
from app import app, create_templates, create_setup_files

if __name__ == '__main__':
    # Create necessary directories and files
    create_templates()
    
    # Get configuration from environment
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Executive Summary Analyzer...")
    print(f"Server running at http://{host}:{port}")
    print(f"Debug mode: {debug}")
    
    # Run the application
    app.run(host=host, port=port, debug=debug)
