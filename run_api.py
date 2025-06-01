#!/usr/bin/env python3
"""
Startup script for the Store Sales Prediction API
"""

from api import app
import uvicorn

if __name__ == "__main__":
    print("Starting Store Sales Prediction API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check endpoint: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )