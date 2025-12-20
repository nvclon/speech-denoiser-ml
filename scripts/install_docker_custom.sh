#!/bin/bash
set -e

# Directory where Docker will store all images and containers
DATA_DIR="/ml/docker-data"

echo "[INFO] Installing Docker..."

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker packages
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[INFO] Stopping Docker to configure data directory..."
sudo systemctl stop docker

echo "[INFO] Creating data directory at $DATA_DIR..."
sudo mkdir -p "$DATA_DIR"
# Copy existing data if any (optional, usually empty on fresh install)
# sudo rsync -aqxP /var/lib/docker/ "$DATA_DIR"

echo "[INFO] Configuring Docker to use $DATA_DIR..."
# Create daemon.json with data-root config
echo "{\"data-root\": \"$DATA_DIR\"}" | sudo tee /etc/docker/daemon.json

echo "[INFO] Starting Docker..."
sudo systemctl start docker

echo "[INFO] Adding user $USER to docker group..."
sudo usermod -aG docker $USER

echo "[INFO] Installing NVIDIA Container Toolkit (required for GPU)..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "----------------------------------------------------------------"
echo "Done! Docker is installed and configured to use $DATA_DIR."
echo "Please LOG OUT and LOG BACK IN to apply group changes."
echo "Then verify with: docker run --rm --gpus all hello-world"
echo "----------------------------------------------------------------"
