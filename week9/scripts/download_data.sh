#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p promedus
asia=https://raw.githubusercontent.com/TensorBFS/TensorInference.jl/935ed86f8e473f523cac9461c45a1f470e74f0f9/examples/asia-network/model.uai
mirror=https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014
curl --fail --location "$asia" --output asia.uai
for suffix in uai uai.evid uai.PR; do
  curl --fail --location "$mirror/PR/Promedus_14.$suffix" --output "promedus/Promedus_14.$suffix"
done
curl --fail --location "$mirror/MAR/Promedus_14.uai.MAR" --output promedus/Promedus_14.uai.MAR
