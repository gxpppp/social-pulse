#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
LOG_DIR="${LOG_DIR:-/app/logs}"
DATABASE_URL="${DATABASE_URL:-sqlite:///./data/sentiment.db}"

echo "========================================"
echo "Sentiment Analyzer - Docker Entry Point"
echo "========================================"
echo "Environment: ${ENVIRONMENT:-development}"
echo "Database: ${DATABASE_URL}"
echo "Redis: ${REDIS_URL}"
echo "========================================"

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for ${service} to be ready..."
    
    while ! nc -z "${host}" "${port}" 2>/dev/null; do
        if [ ${attempt} -ge ${max_attempts} ]; then
            echo "Error: ${service} not available after ${max_attempts} attempts"
            exit 1
        fi
        echo "Attempt ${attempt}/${max_attempts}: ${service} not ready yet, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "${service} is ready!"
}

if [ "${WAIT_FOR_DB}" = "true" ] && echo "${DATABASE_URL}" | grep -q "postgresql"; then
    DB_HOST=$(echo "${DATABASE_URL}" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo "${DATABASE_URL}" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    if [ -n "${DB_HOST}" ] && [ -n "${DB_PORT}" ]; then
        wait_for_service "${DB_HOST}" "${DB_PORT}" "PostgreSQL"
    fi
fi

if [ "${WAIT_FOR_REDIS}" = "true" ] && [ -n "${REDIS_URL}" ]; then
    REDIS_HOST=$(echo "${REDIS_URL}" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
    REDIS_PORT=$(echo "${REDIS_URL}" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    if [ -n "${REDIS_HOST}" ] && [ -n "${REDIS_PORT}" ]; then
        wait_for_service "${REDIS_HOST}" "${REDIS_PORT}" "Redis"
    fi
fi

if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "Running database migrations..."
    if [ -f "/app/sentiment_analyzer/storage/migrations/001_init_schema.sql" ]; then
        if echo "${DATABASE_URL}" | grep -q "postgresql"; then
            DB_HOST=$(echo "${DATABASE_URL}" | sed -n 's/.*@\([^:]*\):.*/\1/p')
            DB_PORT=$(echo "${DATABASE_URL}" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
            DB_NAME=$(echo "${DATABASE_URL}" | sed -n 's/.*\/\([^?]*\).*/\1/p')
            DB_USER=$(echo "${DATABASE_URL}" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
            DB_PASS=$(echo "${DATABASE_URL}" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
            
            export PGPASSWORD="${DB_PASS}"
            psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
                -f /app/sentiment_analyzer/storage/migrations/001_init_schema.sql || true
            unset PGPASSWORD
            echo "PostgreSQL migrations completed."
        else
            echo "SQLite database detected, skipping migrations (auto-created)."
        fi
    else
        echo "No migration files found, skipping."
    fi
fi

if [ "${INIT_DATA}" = "true" ]; then
    echo "Initializing sample data..."
    python -c "
from sentiment_analyzer.storage.database import Database
from sentiment_analyzer.storage.schema import init_database
import asyncio

async def init():
    db = Database('${DATABASE_URL}')
    await init_database(db)
    print('Database initialized successfully.')

asyncio.run(init())
" || echo "Data initialization skipped (may already exist)."
fi

echo "Starting application..."
echo "Command: $@"

exec "$@"
