#!/bin/bash
# WARNING: Make sure you have the necessary permissions to run this script
# by running: chmod +x RUN_THIS_INFERENCE.sh

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in a pipeline from being masked.
set -o pipefail

# default configuration
IMAGE_NAME="intent-classifier-inference-cpu"
IMAGE_TAG="latest"
CONTAINER_NAME="intent-classifier-inference-cpu"
HOST_PORT="8000"
CONTAINER_PORT="8000" # should match Dockerfile EXPOSE/CMD
MODEL_ARTIFACTS_PATH_HOST="./results/BEST_atis_multilabel_xlmr_lora" # relative to script location
INFERENCE_DEVICE="cpu"

info() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Builds and runs the Intent Classification API Docker container"
    echo ""
    echo "Options:"
    echo "  -i, --image-name NAME     Set the Docker image name (default: ${IMAGE_NAME})"
    echo "  -t, --tag TAG             Set the Docker image tag (default: ${IMAGE_TAG})"
    echo "  -c, --container-name NAME Set the container name (default: ${CONTAINER_NAME})"
    echo "  -p, --port HOST_PORT      Set the host port to map to the container's port ${CONTAINER_PORT} (default: ${HOST_PORT})"
    echo "  -m, --model-path PATH     Set the path to the model artifacts directory relative to the script (default: ${MODEL_ARTIFACTS_PATH_HOST})"
    # Add help for other potential args like device
    echo "  -d, --device DEVICE       Set the inference device inside the container (cpu/cuda) (default: ${INFERENCE_DEVICE})"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --port 9000 --model-path ./results/my_other_model"
    exit 0
}


# Loop through arguments and process them
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--image-name)
        IMAGE_NAME="$2"
        shift # past argument
        shift # past value
        ;;
        -t|--tag)
        IMAGE_TAG="$2"
        shift # past argument
        shift # past value
        ;;
        -c|--container-name)
        CONTAINER_NAME="$2"
        shift # past argument
        shift # past value
        ;;
        -p|--port)
        HOST_PORT="$2"
        shift # past argument
        shift # past value
        ;;
        -m|--model-path)
        MODEL_ARTIFACTS_PATH_HOST="$2"
        shift # past argument
        shift # past value
        ;;
        -d|--device)
        INFERENCE_DEVICE="$2"
        shift # past argument
        shift # past value
        ;;
        -h|--help)
        usage
        ;;
        *)    # unknown option
        echo "Unknown option: $1" >&2
        usage
        ;;
    esac
done

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    error "Docker command could not be found. Please install Docker."
fi
if ! docker info &> /dev/null; then
    error "Docker daemon does not seem to be running. Please start Docker."
fi

# make sure if the model path exists relative to the script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ABSOLUTE_MODEL_PATH="${SCRIPT_DIR}/${MODEL_ARTIFACTS_PATH_HOST}"
if [ ! -d "${ABSOLUTE_MODEL_PATH}" ]; then
    error "Model artifacts path not found at: ${ABSOLUTE_MODEL_PATH}"
    info "Please ensure the path '${MODEL_ARTIFACTS_PATH_HOST}' is correct relative to the script or use the --model-path argument."
    exit 1
fi


# Build
info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}..."
info "Using model artifacts from host path: ${MODEL_ARTIFACTS_PATH_HOST}"

# Run docker build from the script's directory
cd "${SCRIPT_DIR}" || exit 1
docker build \
    --build-arg MODEL_ARTIFACTS_PATH="${MODEL_ARTIFACTS_PATH_HOST}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile.inference . # Use current directory as build context

info "Build complete."

# Run
info "Preparing to run container: ${CONTAINER_NAME}..."

# Stop and remove any existing container with the same name
info "Checking for existing container named '${CONTAINER_NAME}'..."
docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
info "Existing container removed (if any)."

info "Starting new container '${CONTAINER_NAME}'..."
info "Mapping host port ${HOST_PORT} to container port ${CONTAINER_PORT}"
info "Setting inference device inside container to: ${INFERENCE_DEVICE}"

# Runing the new container in detached mode
# Use --rm so container is removed automatically when stopped
docker run -d --rm \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    -e API_PORT="${CONTAINER_PORT}" \
    -e INFERENCE_DEVICE="${INFERENCE_DEVICE}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:${IMAGE_TAG}"

info "Container '${CONTAINER_NAME}' started successfully."
echo
info "API Server should be accessible at: http://127.0.0.1:${HOST_PORT}"
info "Check readiness endpoint: curl http://127.0.0.1:${HOST_PORT}/ready"
info "View container logs: docker logs ${CONTAINER_NAME} -f"
info "Stop the container: docker stop ${CONTAINER_NAME}"
echo
