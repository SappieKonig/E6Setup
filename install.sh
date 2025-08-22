#!/bin/bash

set -e

echo "🚀 E6Setup: Setting up ML environment for CIFAR training..."

install_system_deps() {
    echo "📦 Installing system dependencies..."
    sudo apt update
    sudo apt install -y git curl
}

configure_system_settings() {
    echo "⚙️ Configuring system for optimal training performance..."
    
    # Set power profile to performance mode
    if command -v powerprofilesctl &> /dev/null; then
        powerprofilesctl set performance 2>/dev/null && echo "✓ Power profile set to Performance" || echo "Note: Could not set power profile"
    else
        echo "Note: powerprofilesctl not available, skipping power profile"
    fi
    
    # Disable screen blanking and suspend for current session
    if [ "$XDG_SESSION_TYPE" = "x11" ] || [ "$XDG_SESSION_TYPE" = "wayland" ]; then
        # Disable screen blanking
        gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
        gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
        
        # Disable automatic suspend
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true
        
        echo "✓ Display and power settings configured"
    else
        echo "Note: Not in a graphical session, skipping display settings"
    fi
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
    uv sync
}

run_training() {
    echo "🏃 Starting CIFAR training..."
    cd ~/E6Setup
    uv run python main.py
}

main() {
    echo "Starting E6Setup installation..."
    
    install_system_deps
    configure_system_settings
    install_uv
    setup_project
    run_training
    
    echo "✅ E6Setup complete! Check ~/E6Setup for results."
}

main "$@"