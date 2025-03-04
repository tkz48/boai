#!/bin/bash

# Start Sidebolt - Combined Bolt frontend and Sidecar service
# This script starts both components in parallel

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
  echo "Creating .env file from .env.example..."
  cp .env.example .env
  echo "Please update the .env file with your API keys"
fi

# Function to handle cleanup on exit
cleanup() {
  echo "Shutting down services..."
  kill $BOLT_PID $SIDECAR_PID 2>/dev/null
  exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Start Sidecar service
echo "Starting Sidecar service..."
cd sidecar
cargo run --bin webserver &
SIDECAR_PID=$!
cd ..

# Wait for Sidecar to initialize
echo "Waiting for Sidecar to initialize..."
sleep 5

# Start Bolt frontend
echo "Starting Bolt frontend..."
node pre-start.cjs && remix vite:dev &
BOLT_PID=$!

echo "Sidebolt is running!"
echo "- Frontend: http://localhost:5173"
echo "- Sidecar API: http://localhost:3000"
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $BOLT_PID $SIDECAR_PID