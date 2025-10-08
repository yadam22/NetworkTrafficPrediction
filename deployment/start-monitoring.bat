@echo off
setlocal

echo Starting Network Traffic Monitoring Stack:

REM Create necessary directories
if not exist "monitoring\prometheus" mkdir monitoring\prometheus
if not exist "monitoring\grafana\provisioning\datasources" mkdir monitoring\grafana\provisioning\datasources
if not exist "monitoring\grafana\provisioning\dashboards" mkdir monitoring\grafana\provisioning\dashboards
if not exist "monitoring\grafana\dashboards" mkdir monitoring\grafana\dashboards

REM Check if configuration files exist
if not exist "monitoring\prometheus\prometheus.yml" (
    echo Error: Prometheus configuration not found!
    echo Please ensure all monitoring configuration files are in place.
    exit /b 1
)

REM Start monitoring stack
docker-compose -f docker/docker-compose.monitoring.yml up -d

echo Waiting for services to start:
timeout /t 10 /nobreak > nul

REM Check service health
echo Checking service status:

REM Check Prometheus
curl -s http://localhost:9090/-/healthy > nul 2>&1
if %errorlevel% == 0 (
    echo  Prometheus is running at http://localhost:9090
) else (
    echo  Prometheus is not healthy
)

REM Check Grafana
curl -s http://localhost:3000/api/health > nul 2>&1
if %errorlevel% == 0 (
    echo  Grafana is running at http://localhost:3000
    echo   Username: admin ^(or value from .env^)
    echo   Password: Check .env file ^(default: admin123^)
) else (
    echo  Grafana is not healthy
)

REM Check API
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% == 0 (
    echo  API is running with metrics at http://localhost:8000/metrics
) else (
    echo  API is not healthy
)

echo.
echo Monitoring stack started successfully!
echo.
echo Access points:
echo   - Grafana: http://localhost:3000
echo   - Prometheus: http://localhost:9090
echo   - API Metrics: http://localhost:8000/metrics
echo.
echo Default Grafana credentials:
echo   Username: admin
echo   Password: %GRAFANA_PASSWORD%
echo.
echo To view logs: docker-compose -f docker/docker-compose.monitoring.yml logs -f
echo To stop: docker-compose -f docker/docker-compose.monitoring.yml down

endlocal