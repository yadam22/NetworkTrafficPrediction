#!/bin/bash

set -e

echo "Starting Network Traffic Monitoring Stack:"

# Create necessary directories
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# Check if configuration files exist
if [ ! -f "monitoring/prometheus/prometheus.yml" ]; then
    echo "Error: Prometheus configuration not found!"
    echo "Please ensure all monitoring configuration files are in place."
    exit 1
fi

# Start monitoring stack
docker-compose -f docker/docker-compose.monitoring.yml up -d

echo "Waiting for services to start:"
sleep 10

# Check service health
echo "Checking service status:"

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo " Prometheus is running at http://localhost:9090"
else
    echo " Prometheus is not healthy"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "Grafana is running at http://localhost:3000"
    echo "  Username: admin (or value from .env)"
    echo "  Password: Check .env file (default: admin123)"
else
    echo " Grafana is not healthy"
fi

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo " API is running with metrics at http://localhost:8000/metrics"
else
    echo " API is not healthy"
fi

echo ""
echo "Monitoring stack started successfully!"
echo ""
echo "Access points:"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
echo "  - API Metrics: http://localhost:8000/metrics"
echo ""
echo "Default Grafana credentials:"
echo "  Username: admin"
echo "  Password: Password: ${GRAFANA_PASSWORD}"
echo ""
echo "To view logs: docker-compose -f docker/docker-compose.monitoring.yml logs -f"
echo "To stop: docker-compose -f docker/docker-compose.monitoring.yml down"