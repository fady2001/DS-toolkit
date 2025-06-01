#!/bin/bash

# Store Sales Prediction API - Docker Deployment Script

set -e  # Exit on any error

# Configuration
IMAGE_NAME="store-sales-api"
IMAGE_TAG="latest"
CONTAINER_NAME="store-sales-container"

echo "🚀 Store Sales Prediction API - Docker Deployment"
echo "=================================================="

# Function to display usage
usage() {
    echo "Usage: $0 [build|run|stop|restart|logs|clean]"
    echo ""
    echo "Commands:"
    echo "  build    - Build the Docker image"
    echo "  run      - Run the container"
    echo "  stop     - Stop the container"
    echo "  restart  - Restart the container"
    echo "  logs     - Show container logs"
    echo "  clean    - Clean up containers and images"
    exit 1
}

# Build Docker image
build_image() {
    echo "📦 Building Docker image..."
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    echo "✅ Image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Run container
run_container() {
    echo "🏃 Starting container..."
    
    # Stop existing container if running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        echo "⚠️  Stopping existing container..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi
    
    # Run new container
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p 8000:8000 \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    echo "✅ Container started successfully!"
    echo "🌐 API available at: http://localhost:8000"
    echo "📖 Documentation at: http://localhost:8000/docs"
    echo "❤️  Health check at: http://localhost:8000/health"
}

# Stop container
stop_container() {
    echo "🛑 Stopping container..."
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        echo "✅ Container stopped and removed"
    else
        echo "ℹ️  Container is not running"
    fi
}

# Show logs
show_logs() {
    echo "📋 Container logs:"
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        echo "❌ Container is not running"
    fi
}

# Clean up
cleanup() {
    echo "🧹 Cleaning up..."
    
    # Stop and remove container
    if docker ps -a -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
    
    # Remove image
    if docker images -q ${IMAGE_NAME}:${IMAGE_TAG} | grep -q .; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
    fi
    
    echo "✅ Cleanup completed"
}

# Main script logic
case "${1:-}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        build_image
        run_container
        ;;
    logs)
        show_logs
        ;;
    clean)
        cleanup
        ;;
    *)
        usage
        ;;
esac

echo "🎉 Operation completed!"