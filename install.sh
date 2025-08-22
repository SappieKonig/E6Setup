#!/bin/bash

set -e

echo "🚀 E6Setup: Setting up ML environment for CIFAR training..."

install_system_deps() {
    echo "📦 Installing system dependencies..."
    sudo apt update
    sudo apt install -y git curl
}

install_uv() {
    echo "🐍 Installing uv (Python environment manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
}

setup_project() {
    echo "📁 Setting up E6Setup project..."
    cd ~
    if [ -d "E6Setup" ]; then
        echo "Directory E6Setup already exists, removing..."
        rm -rf E6Setup
    fi
    
    git clone https://github.com/SappieKonig/E6Setup.git
    cd E6Setup
    
    echo "🔧 Creating Python environment and installing dependencies..."
    ~/.local/bin/uv sync
}

run_training() {
    echo "🏃 Starting CIFAR training..."
    cd ~/E6Setup
    ~/.local/bin/uv run python main.py
}

main() {
    echo "Starting E6Setup installation..."
    
    install_system_deps
    install_uv
    setup_project
    run_training
    
    echo "✅ E6Setup complete! Check ~/E6Setup for results."
}

main "$@"