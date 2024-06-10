#!/bin/bash

# Function to print informative messages
log_info() {
    echo "[INFO] $1"
}

# Function to print error messages and exit the script
log_error() {
    echo "[ERROR] $1"
    exit 1
}

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    log_error "conda is not installed. Please install conda and try again."
fi

# Check if curl is installed
if ! command -v curl &> /dev/null
then
    log_error "curl is not installed. Please install curl and try again."
fi

# Set variables
ENV_NAME="SentenceClassification"
REQUIREMENTS_FILE="setup/requirements.yaml"

log_info "Activating base conda environment..."
conda activate base || log_error "Failed to activate base conda environment."

log_info "Removing existing $ENV_NAME conda environment..."
conda remove -n $ENV_NAME --all --yes || log_error "Failed to remove existing $ENV_NAME conda environment."

log_info "Creating new $ENV_NAME conda environment..."
conda env create -f $REQUIREMENTS_FILE || log_error "Failed to create $ENV_NAME conda environment."

log_info "Activating $ENV_NAME conda environment..."
conda activate $ENV_NAME || log_error "Failed to activate $ENV_NAME conda environment."

log_info "Installing ollama..."
curl -fsSL https://ollama.com/install.sh | sh || log_error "Failed to install ollama."

log_info "Pulling llama3 model..."
ollama pull llama3 || log_error "Failed to pull llama3 model."

log_info "Script execution completed successfully."