#!/bin/bash
set -e

NODES_DIR="/comfyui/custom_nodes"
mkdir -p "$NODES_DIR"

install_node() {
    local repo=$1
    local name=${2:-$(basename "$repo" .git)}
    local target="$NODES_DIR/$name"

    if [ ! -d "$target/.git" ]; then
        echo "[nodes] Installing $name..."
        git clone "$repo" "$target"
        if [ -f "$target/requirements.txt" ]; then
            pip install --no-cache-dir -r "$target/requirements.txt"
        fi
    else
        echo "[nodes] $name already present — skipping."
    fi
}

install_node "https://github.com/city96/ComfyUI-GGUF"
install_node "https://github.com/drphero/ComfyUI-FASHN-VTON"
install_node "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
install_node "https://github.com/kijai/ComfyUI-KJNodes"
