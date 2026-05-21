#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_common.sh"

PRETRAINED_DIR=${PRETRAINED_DIR:-$DATA_ROOT/pretrained}
mkdir -p "$PRETRAINED_DIR"

download_if_missing() {
  local url=$1
  local output=$2

  if [[ -s "$output" ]]; then
    echo "exists: $output"
    return
  fi

  echo "download: $url"
  wget -c -O "$output" "$url"
}

download_if_missing \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
  "$PRETRAINED_DIR/edm-cifar10-32x32-cond-vp.pkl"

download_if_missing \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
  "$PRETRAINED_DIR/edm-cifar10-32x32-uncond-vp.pkl"

download_if_missing \
  https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz \
  "$PRETRAINED_DIR/cifar10-32x32.npz"

echo "ready: $PRETRAINED_DIR"

