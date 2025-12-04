#!/bin/bash

# CUDA Installation Script for WSL with RTX 4080
set -e

echo "=================================================="
echo "CUDA Installation Script for WSL/RTX 4080"
echo "=================================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script as root/sudo"
    exit 1
fi

# Function to print status messages
print_status() {
    echo -e "\n\033[1;34m[$1]\033[0m $2"
}

# Function to print success messages
print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

# Function to print error messages
print_error() {
    echo -e "\033[1;31m✗ $1\033[0m"
}

# Update system
print_status "STEP 1" "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_success "System updated successfully"

# Install dependencies
print_status "STEP 2" "Installing dependencies..."
sudo apt install -y build-essential wget software-properties-common
print_success "Dependencies installed"

# Download and install CUDA keyring
print_status "STEP 3" "Downloading CUDA repository keyring..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
print_success "CUDA keyring installed"

# Update package list with CUDA repository
print_status "STEP 4" "Updating package list with CUDA repository..."
sudo apt update
print_success "Package list updated"

# Install CUDA toolkit
print_status "STEP 5" "Installing CUDA toolkit (this may take a while)..."
sudo apt install -y cuda-toolkit-12-4
print_success "CUDA toolkit installed"

# Set up environment variables
print_status "STEP 6" "Configuring environment variables..."
{
    echo ''
    echo '# CUDA Paths'
    echo 'export PATH=/usr/local/cuda/bin:$PATH'
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    echo 'export CUDA_HOME=/usr/local/cuda'
} >> ~/.bashrc

# Also add to current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

print_success "Environment variables configured"

# Create verification script
print_status "STEP 7" "Creating verification script..."
cat > ~/verify_cuda_install.sh << 'EOF'
#!/bin/bash
echo "=== CUDA Installation Verification ==="

echo -e "\n1. Checking nvcc version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH"
fi

echo -e "\n2. Checking nvidia-smi:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available"
fi

echo -e "\n3. Checking CUDA directory:"
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA installed at: /usr/local/cuda"
    ls -la /usr/local/cuda
else
    echo "CUDA directory not found"
fi

echo -e "\n4. Testing CUDA compilation:"
cat > /tmp/test_cuda.cu << 'TESTEOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\\n", i, prop.name);
        printf("  Compute Capability: %d.%d\\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    }
    return 0;
}
TESTEOF

if command -v nvcc &> /dev/null; then
    nvcc /tmp/test_cuda.cu -o /tmp/test_cuda
    if [ -f "/tmp/test_cuda" ]; then
        echo "Compilation successful, running test:"
        /tmp/test_cuda
    else
        echo "Compilation failed"
    fi
else
    echo "nvcc not available for compilation test"
fi

echo -e "\n=== Verification Complete ==="
EOF

chmod +x ~/verify_cuda_install.sh
print_success "Verification script created"

# Cleanup
print_status "STEP 8" "Cleaning up..."
rm -f cuda-keyring_1.0-1_all.deb
print_success "Cleanup completed"

print_status "COMPLETE" "CUDA installation finished!"
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Verify installation by running: ./verify_cuda_install.sh"
echo "3. Ensure you have the latest NVIDIA drivers on Windows"
echo ""
echo "If nvidia-smi doesn't work, make sure:"
echo "- You have Windows 11 with WSL 2"
echo "- Latest NVIDIA drivers are installed on Windows"
echo "- Your RTX 4080 is properly detected in Windows"

# Reload bashrc for current session
source ~/.bashrc

# Run verification
echo ""
read -p "Run verification script now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./verify_cuda_install.sh
fi