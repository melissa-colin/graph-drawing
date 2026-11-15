#!/usr/bin/env bash
# Downloads the code of the two learned baselines, at the exact versions used for our results, into benchmark/ext/
# (not versioned). Run once from the project root:  bash benchmark/setup_baselines.sh
# - CoRe-GD (ICLR 2024): its repo also ships the pretrained Rome checkpoints of CoRe-GD and of DeepGD.
# - DeepGD: the code of the model that matches those checkpoints (an older version of the DeepGD repo).
set -euo pipefail
EXT="$(dirname "$0")/ext"
mkdir -p "$EXT"
if [ ! -d "$EXT/CoRe-GD" ]; then
    git clone -q https://github.com/floriangroetschla/CoRe-GD.git "$EXT/CoRe-GD"
    git -C "$EXT/CoRe-GD" checkout -q 30f7067306e37100a80c7034a52902292905898a
fi
if [ ! -d "$EXT/deepgd" ]; then
    git clone -q https://github.com/yolandalalala/DeepGD-Demo.git "$EXT/deepgd"
    git -C "$EXT/deepgd" checkout -q 329cadf788fc9d205361023c4a9c3537da21e171
    # Recent PyTorch Geometric checks the propagate_type comment against the arguments actually passed.
    sed -i 's/# propagate_type: (x: OptPairTensor, edge_attr: OptTensor)$/# propagate_type: (x: OptPairTensor, edge_attr: OptTensor, edge_weight: OptTensor)/' \
        "$EXT/deepgd/deepgd/nn/modules/weighted_nnconv.py"
fi
echo "baselines ready in $EXT"
