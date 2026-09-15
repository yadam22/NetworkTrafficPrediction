# Network Traffic Prediction with LLMs 📉📈

This research project, that involves transformer models with hierarchical tokenization and parameter-efficient fine-tuning, explores the application of Large Language Models (LLMs) and traditional methods for network traffic prediction.

---

## Overview

This project explores **network traffic forecasting** using modern ML approaches, comparing transformer models against deep learning methods like **LSTM** and traditional statistical methods like **ARIMA**.

- **Hierarchical tokenization system**: maps traffic to a **136-token vocabulary** with **93.4% sequence reduction**
- **LoRA (Low-Rank Adaptation)**: used for parameter-efficient fine-tuning

---

## Key Results

- **Transformer Model**: MAE = `0.0053` on 30-second windows (**98.9% improvement over LSTM**)
- **LSTM Model**: R² = `0.915`, inference time = `0.279ms` → **optimal for real-time**
- **ARIMA**: serious failures, errors > `3.67M`
- **Accuracy-Latency Tradeoff**: Transformers excel in **batch analysis**, LSTM is **best for real-time**

---

## Architecture

### Docker Setup

Base container for GPU-accelerated processing:

```bash
docker run -dit --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "%cd%":/workspace -w /workspace --name network_predict \
  nvcr.io/nvidia/rapidsai/base:25.08-cuda12.0-py3.11
```

### Container Configuration

- **Base Image**: `nvcr.io/nvidia/rapidsai/base:25.08-cuda12.0-py3.11`
- **CUDA Support**: CUDA 12.0 with GPU acceleration
- **Memory**: `--shm-size=1g` for large dataset handling
- **System Limits**: `memlock` and `stack` adjusted for optimal ML performance
- **Data & Models**: located in `code/src/data` and `code/src/models`

---

## Project Structure

```plaintext
├── code/
│   └── src/
│       ├── data/          # Dataset and preprocessing
│       └── models/        # Model implementations
├── deployment/
│   ├── docker/
│   │   └── docker-compose.monitoring.yml
│   ├── start-monitoring.bat  # Windows monitoring setup
│   └── start-monitoring.sh   # Linux monitoring setup
├── run_api.py
└── run_api.bat
```

---

## Quick Start

### 1. GPU Container Setup first cd to the project root and ensure Docker Desktop (or Docker) is running.

**Linux:**

```bash
# Start the RAPIDS container in a linux terminal
docker run -dit --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":/workspace -w /workspace --name network_predict \
  nvcr.io/nvidia/rapidsai/base:25.08-cuda12.0-py3.11
```

#### Windows:

```bash
# Start the RAPIDS container in a powershell terminal
docker run -dit --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 `
  -v "${PWD}:/workspace" -w /workspace --name network_predict `
  nvcr.io/nvidia/rapidsai/base:25.08-cuda12.0-py3.11

# In CMD terminal run
docker run -dit --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v %cd%:/workspace -w /workspace --name network_predict nvcr.io/nvidia/rapidsai/base:25.08-cuda12.0-py3.11

```

#### Once done

```bash
docker start -ai container_ID

cd /workspace

# You first need to run the requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-prod.txt

# Run any of the files with !python3 followed by the relative directory of the file you want to execute with forward slashes (as docker runs in a linux environment)
!python3 code/src/data/...
!python3 code/src/models/...
```

### 2. API Deployment

**Local (Windows):**

```bash
run_api.bat
```

**Or if not in a windows environment**

```bash
python run_api.py
```

**Docker Deployment:**

```bash
docker-compose -f deployment/docker/docker-compose.yml build api
docker-compose -f deployment/docker/docker-compose.yml up -d
# Or with monitoring
docker-compose -f deployment/docker/docker-compose.monitoring.yml build api
docker-compose -f deployment/docker/docker-compose.monitoring.yml up -d
```

### 3. Monitoring Setup

**Windows:**

```bash
deployment\start-monitoring.bat
```

**Linux:**

```bash
deployment/start-monitoring.sh
```

**Docker (Full Stack):**

```bash
docker-compose -f deployment/docker/docker-compose.monitoring.yml
docker-compose -f deployment/docker/docker-compose.monitoring.yml up -d
```

---

## Service Ports

- **API**: `8000`
- **Grafana**: `3000`
- **Prometheus**: `9090`

---

## Docker Management

**Rebuild API Container**

```bash
docker-compose -f deployment/docker/docker-compose.monitoring.yml build api
```

**Restart Stack**

```bash
docker-compose -f deployment/docker/docker-compose.monitoring.yml down
docker-compose -f deployment/docker/docker-compose.monitoring.yml up -d
```

**Monitor Logs**

```bash
docker-compose -f deployment/docker/docker-compose.monitoring.yml logs -f api
docker-compose -f deployment/docker/docker-compose.monitoring.yml logs -f prometheus
docker-compose -f deployment/docker/docker-compose.monitoring.yml logs -f grafana
docker-compose -f deployment/docker/docker-compose.monitoring.yml logs -f node-exporter
```

**Restart**

```bash
docker-compose -f deployment/docker/docker-compose.monitoring.yml restart api
docker-compose -f deployment/docker/docker-compose.monitoring.yml restart prometheus
docker-compose -f deployment/docker/docker-compose.monitoring.yml restart grafana
docker-compose -f deployment/docker/docker-compose.monitoring.yml restart node-exporter
```

---

## Important Notes

### PyTorch DataLoader Configuration

- **Notebooks (Windows)**: `num_workers=0` → avoids pickling issues
- **Docker Container**: `num_workers=4` → multiprocessing works properly in RAPIDS container

> PyTorch multiprocessing DataLoader has compatibility issues on Windows but it functions properly inside the GPU accelerated Docker environment.

---

## GitLab CI/CD Pipeline

GitLab CI/CD pipeline with Docker runners for automated testing and GCP deployment.

### Required GitLab Variables

| Variable                  | Description                             | Example                       |
| ------------------------- | --------------------------------------- | ----------------------------- |
| `GCP_SERVICE_ACCOUNT_KEY` | Base64 encoded GCP service account JSON | `eyJ0eXBlIjogInNlcnZpY2Vf...` |
| `GCP_PROJECT_ID`          | Google Cloud Project ID                 | `my-project-123456`           |
| `GCP_ARTIFACT_REGISTRY`   | Artifact Registry host                  | `europe-west2-docker.pkg.dev` |
| `GCP_REPOSITORY_NAME`     | Repository name                         | `my-docker-repo`              |

### Encode Service Account Key

#### Linux

```bash
base64 -w 0 service-account-key.json
```

---

### Performance Characteristics

| Model       | MAE    | Inference Time | Use Case                      |
| ----------- | ------ | -------------- | ----------------------------- |
| Transformer | 0.0053 | 13.91–44.58 ms | Batch analysis, high accuracy |
| LSTM        | 0.490  | 0.279 ms       | Real-time prediction          |
| ARIMA       | >3.67M | -              | Not suitable                  |

### GitLab Runner Setup

**Linux**

```bash
# Create volume and start runner
docker volume create gitlab-runner-config

docker run -d --name gitlab-runner --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v gitlab-runner-config:/etc/gitlab-runner \
  gitlab/gitlab-runner:latest

# Register runner
docker exec -it gitlab-runner gitlab-runner register
```

**Windows**

```bash
# Create volume and start runner
docker volume create gitlab-runner-config

# For Powershell run
docker run -d --name gitlab-runner --restart always `
  -v \\.\pipe\docker_engine:\\.\pipe\docker_engine `
  -v gitlab-runner-config:/etc/gitlab-runner `
  gitlab/gitlab-runner:latest
# For CMD run
docker run -d --name gitlab-runner --restart always -v \\.\pipe\docker_engine:\\.\pipe\docker_engine -v gitlab-runner-config:/etc/gitlab-runner gitlab/gitlab-runner:latest

# Register runner
docker exec -it gitlab-runner gitlab-runner register
```

---

## Deployment Architecture

- **Framework**: FastAPI (modular design)
- **Containerization**: Docker with GPU support
- **CI/CD**: Automated deployment pipeline
- **Cloud**: Google Cloud Run deployment
- **Monitoring**: Prometheus + Grafana stack

---

## Research Context

- **Transformer Models**: For superior accuracy and is ideal for batch processing
- **LSTM Models**: Lower accuracy but optimal for real-time prediction
- **Hybrid Architectures**: Combining transformer and LSTM can allow for a high accuracy real-world deployment

**Dataset**: CAIDA backbone traffic data
**Tokenization**: Hierarchical system, 136-token vocabulary
**Fine-tuning**: LoRA → only 18.51% of parameters are trainable

---

## API Performance

- Mean response time: **5.26s** (improvements required for production)
- Suitable for **batch processing and analysis workflows**
- **Real-time applications** should use LSTM

---

## Getting Help

- **GPU Acceleration**: Ensure NVIDIA drivers and Docker GPU support
- **Memory Errors**: Confirm `--shm-size=1g` is set
- **Port Conflicts**: Check availability of `8000`, `3000`, `9090`
- **Required Software**: Docker Desktop, NVIDIA Docker Runtime, Python 3.11, Git with download links
- **System Requirements**: GPU specs, RAM, storage and OS requirements
- **Environment help**: Verify the NVIDIA drivers, Docker GPU support and CUDA compatibility before running GPU workloads
- **Verification Commands**: To test that everything is installed correctly
- **License**: This project is distributed under the repository’s license, check LICENSE for permitted uses and attribution.

---
