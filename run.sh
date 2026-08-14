#!/bin/bash

# Try to kill any existing processes on ports 8000 and 5173 to prevent conflicts
fuser -k 8000/tcp 2>/dev/null || true

echo "Memulai Backend API..."
source venv/bin/activate
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo ""
echo "=================================================="
echo "Aplikasi Portal Prediksi AI berjalan!"
echo "Backend: http://localhost:8000"
echo "Tekan Ctrl+C untuk menghentikan semua server."
echo "=================================================="

# Wait for user interrupt
trap "echo 'Menghentikan server...'; kill $BACKEND_PID 2>/dev/null; kill $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait


