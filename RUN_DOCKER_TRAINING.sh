#!/bin/bash
# WARNING: Make sure you have the necessary permissions to run this script
# by running: chmod +x RUN_THIS_INFERENCE.sh

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in a pipeline from being masked.
set -o pipefail

# --- Default Configuration ---
IMAGE_NAME="seq-classifier-trainer"
IMAGE_TAG="latest"
CONTAINER_NAME_PREFIX="seq-classifier-trainer" # Base name, will append run type and timestamp
HOST_DATA_PATH="./coding_task/data/atis" # Relative path to data from script location
HOST_OUTPUT_BASE_PATH="./results"        # Base dir for output, run-specific dir created inside
DEFAULT_RUN_TYPE="multiclass"             # Default training type if none specified
INTERACTIVE_MODE="-it"                   # Run interactively by default (-it), use "-d" for detached


info() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

usage() {
    echo "Usage: $0 [OPTIONS] [RUN_TYPE]"
    echo "Builds the training Docker image and runs a training container."
    echo ""
    echo "Arguments:"
    echo "  RUN_TYPE              Type of training run ('multiclass' or 'multilabel'). Optional, defaults to '${DEFAULT_RUN_TYPE}'."
    echo ""
    echo "Options:"
    echo "  -i, --image-name NAME     Set the Docker image name (default: ${IMAGE_NAME})"
    echo "  -t, --tag TAG             Set the Docker image tag (default: ${IMAGE_TAG})"
    echo "  -d, --data-path PATH      Host path to the data directory (relative to script location) (default: ${HOST_DATA_PATH})"
    echo "  -o, --output-path PATH    Host path for the base output directory (relative to script location) (default: ${HOST_OUTPUT_BASE_PATH})"
    echo "  -n, --container-name NAME Base name for the container (default: ${CONTAINER_NAME_PREFIX})"
    echo "      --detached            Run container in detached mode instead of interactive"
    echo "  -h, --help                Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 multilabel -o ./custom_results --image-name my-trainer --tag v2"
    echo "  $0 # Runs default 'multiclass' training"
    exit 0
}

# Parse options first
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--image-name)
        IMAGE_NAME="$2"
        shift 2 # past argument and value
        ;;
        -t|--tag)
        IMAGE_TAG="$2"
        shift 2
        ;;
        -d|--data-path)
        HOST_DATA_PATH="$2"
        shift 2
        ;;
        -o|--output-path)
        HOST_OUTPUT_BASE_PATH="$2"
        shift 2
        ;;
        -n|--container-name)
        CONTAINER_NAME_PREFIX="$2"
        shift 2
        ;;
        --detached)
        INTERACTIVE_MODE="-d"
        shift 1 # past argument
        ;;
        -h|--help)
        usage
        ;;
        -*) # Unknown option
        echo "Unknown option: $1" >&2
        usage
        ;;
        *) # Positional argument (RUN_TYPE)
        # Break loop once positional arguments start
        break
        ;;
    esac
done

# Assign positional argument (if provided) to RUN_TYPE
RUN_TYPE="${1:-$DEFAULT_RUN_TYPE}" # Use $1 if set, otherwise default

# Validate RUN_TYPE
if [[ "$RUN_TYPE" != "multiclass" && "$RUN_TYPE" != "multilabel" ]]; then
    error "Invalid RUN_TYPE: '${RUN_TYPE}'. Must be 'multiclass' or 'multilabel'."
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    error "Docker command could not be found. Please install Docker."
fi
if ! docker info &> /dev/null; then
    error "Docker daemon does not seem to be running. Please start Docker."
fi

# Check if the host data path exists relative to the script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ABSOLUTE_HOST_DATA_PATH="${SCRIPT_DIR}/${HOST_DATA_PATH}"
if [ ! -d "${ABSOLUTE_HOST_DATA_PATH}" ]; then
    error "Host data path not found at: ${ABSOLUTE_HOST_DATA_PATH}"
    info "Please ensure the path '${HOST_DATA_PATH}' is correct relative to the script or use the --data-path argument."
    exit 1
fi

# Create unique output directory on host
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOST_OUTPUT_PATH="${SCRIPT_DIR}/${HOST_OUTPUT_BASE_PATH}/train_${RUN_TYPE}_${TIMESTAMP}"
info "Creating host output directory: ${HOST_OUTPUT_PATH}"
mkdir -p "${HOST_OUTPUT_PATH}" || error "Failed to create output directory."

# Define unique container name
CONTAINER_NAME="${CONTAINER_NAME_PREFIX}-${RUN_TYPE}-${TIMESTAMP}"

# Build
info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}..."

# Run docker build from the script's dir
cd "${SCRIPT_DIR}" || exit 1 # Change to script dir just in case, exit if fail
docker build \
    -f Dockerfile.train \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile.training . # Use current directory as build context

info "Build complete."

# Run
info "Preparing to run training container: ${CONTAINER_NAME}..."
info "Run Type: ${RUN_TYPE}"
info "Host Data Path (RO): ${ABSOLUTE_HOST_DATA_PATH} -> /app/data"
info "Host Output Path (RW): ${HOST_OUTPUT_PATH} -> /app/output"

# Run container
info "Starting container '${CONTAINER_NAME}'..."
docker run --rm ${INTERACTIVE_MODE} \
    --gpus all \
    -v "${ABSOLUTE_HOST_DATA_PATH}:/app/data:ro" \
    -v "${HOST_OUTPUT_PATH}:/app/output" \
    -v "${HOST_OUTPUT_PATH}/logs:/app/logs" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${RUN_TYPE}"

info "Training container '${CONTAINER_NAME}' finished."
info "Output saved to host path: ${HOST_OUTPUT_PATH}"
echo
