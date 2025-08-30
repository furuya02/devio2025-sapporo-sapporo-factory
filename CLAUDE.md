# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the AHIRU (Duck) Factory project, a DevIO 2025 Sapporo demonstration combining IoT edge computing with cloud infrastructure. The project simulates a duck factory quality control system using:

- **Jetson Edge Device**: Computer vision-based duck quality inspection using YOLO segmentation, embedding models, and servo control
- **AWS Cloud Infrastructure**: CDK-deployed Lambda functions providing Grafana JSON API for factory metrics visualization

## Repository Structure

### CDK Directory (`/CDK/`)
AWS Cloud Development Kit project for cloud infrastructure:
- **Main Stack**: `lib/cdk-stack.ts` - Defines DuckFactoryStack with Lambda and Function URL
- **Lambda Function**: `lambda/json-api/index.ts` - Grafana JSON API endpoints for factory metrics
- **Entry Point**: `bin/cdk.ts` - CDK app initialization

### Jetson Directory (`/Jetson/`)
NVIDIA Jetson edge computing components:
- **Docker Environment**: Custom Docker image based on `dustynv/l4t-pytorch:r36.4.0`
- **Python Application**: Computer vision pipeline for duck quality inspection
- **Documentation**: Step-by-step assembly and testing guides

## Common Development Commands

### CDK Development
```bash
cd CDK
npm run build          # Compile TypeScript
npm run watch          # Watch mode compilation
npm run test           # Run Jest tests
npm run deploy         # Deploy to AWS without approval
npx cdk diff           # Compare deployed vs current
npx cdk synth          # Generate CloudFormation template
npx cdk destroy        # Remove AWS resources
```

### Jetson Development
```bash
cd Jetson/home
./docker-build.sh      # Build custom Docker image
./docker-run.sh        # Run container with GPU, camera, and GUI support
# Inside container:
python3 index.py       # Main application loop
python3 check_iot.py   # IoT functionality test
```

### Python Validation Commands (Inside Jetson Container)
```bash
python3 check_torch.py      # PyTorch GPU acceleration test
python3 check_webcam.py     # Camera functionality test
python3 check_inference.py  # YOLO model segmentation test
python3 check_embedding.py  # Similarity scoring test
python3 check_sensor.py     # GPIO sensor interface test
python3 check_servo.py      # Servo motor control test
python3 check_center.py     # Duck centering algorithm test
python3 check_mask.py       # Image masking functionality test
```

## Architecture Details

### Cloud Infrastructure (CDK)
- **Runtime**: Node.js 18 Lambda with TypeScript
- **API**: Grafana JSON API compatible endpoints (`/metrics`, `/query`)
- **Metrics**: Temperature, humidity, production count, atmospheric pressure
- **Access**: Public Function URL with CORS enabled
- **Testing**: Jest with TypeScript support

### Edge Computing (Jetson)
- **Base Image**: L4T PyTorch container for NVIDIA Jetson
- **Computer Vision**: YOLO segmentation model (`best.pt`)
- **Quality Check**: Embedding-based similarity comparison with baseline duck image
- **Hardware Control**: Servo motor for reject mechanism, GPIO sensors
- **Cloud Integration**: AWS IoT Core for telemetry, S3 for image storage

### Key Python Modules
- `duck_centerer.py`: YOLO-based duck detection and centering
- `embedding.py`: Feature extraction and similarity scoring
- `sensor.py`: GPIO sensor interface
- `servo.py`: Servo motor control for reject mechanism
- `index.py`: Main application loop integrating all components

## Testing and Validation

### CDK Tests
- Unit tests in `/CDK/test/` directory
- Run with `npm test` or `npm test -- --watch`
- TypeScript compilation validation

### Jetson Validation Scripts
All validation scripts are located in `Jetson/home/src/` and should be run inside the Docker container:
- `check_torch.py`: PyTorch GPU acceleration test
- `check_webcam.py`: Camera functionality test  
- `check_inference.py`: YOLO model segmentation test
- `check_embedding.py`: Similarity scoring test
- `check_sensor.py`: GPIO sensor interface test
- `check_servo.py`: Servo motor control test
- `check_center.py`: Duck centering algorithm test
- `check_mask.py`: Image masking functionality test

## Development Environment Setup

### Prerequisites
- AWS CLI configured with appropriate permissions
- Docker for Jetson container development
- NVIDIA Jetson device with camera and servo hardware
- Node.js 18+ for CDK development

### Key Configuration
- **TypeScript**: ES2022 target with strict mode enabled
- **Docker**: GPU runtime with X11 forwarding for GUI applications
- **AWS Permissions**: IoT Core, S3, CloudFormation deployment rights
- **Hardware**: USB camera on `/dev/video0`, GPIO pins for sensors/servos
- **Models**: YOLO segmentation model (`best.pt`) for duck detection
- **Docker Base**: `dustynv/l4t-pytorch:r36.4.0` for NVIDIA Jetson

## Development Workflows

### Setting up Jetson Development Environment
1. Copy source files to Jetson device: `scp -r Jetson/home/* user@jetson-ip:~/project/`
2. Build Docker image: `./docker-build.sh`
3. Run with hardware access: `./docker-run.sh`
4. Validate components using check scripts before running main application

### CDK Development Cycle
1. Make infrastructure changes in `lib/cdk-stack.ts`
2. Update Lambda function in `lambda/json-api/index.ts`
3. Run tests: `npm test`
4. Preview changes: `npx cdk diff`
5. Deploy: `npm run deploy`

## Hardware Requirements
- **Jetson Device**: NVIDIA Jetson with GPU acceleration support
- **Camera**: USB camera compatible with `/dev/video0`
- **Servo Motor**: Connected via GPIO pins for reject mechanism
- **Sensors**: GPIO-connected sensors for production line monitoring

The system demonstrates end-to-end IoT edge-to-cloud architecture with real-time computer vision processing and cloud-based monitoring infrastructure.