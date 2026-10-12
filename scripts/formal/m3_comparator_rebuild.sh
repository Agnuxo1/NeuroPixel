#!/usr/bin/env bash
# M3: rebuild and run Comparator for DirectCrouzeix and CourtadeKumar (prepared; NOT executed).
# Requires owner approval for the downloads below. All heavy files live on E:, never in the WSL disk on C:.
set -euo pipefail
WORK=/mnt/e/Proyecto-concursos/NeuroPixel-lean
OUT=/mnt/d/PROJECTS/NeuroPixel-Completion-20261012/results/research/M3_comparator
MATH_COMMIT=fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb
TOOLCHAIN=leanprover/lean4:v4.34.1
export ELAN_HOME=$WORK/elan PATH=$WORK/elan/bin:$WORK/bin:$PATH
mkdir -p "$WORK/bin" "$OUT"
[ -d "$OUT/run" ] && { echo "refusing to overwrite $OUT/run"; exit 2; }
mkdir "$OUT/run"
# 1. Lean toolchain manager and pinned toolchain (download: elan installer, toolchain ~0.3 GB).
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --no-modify-path --default-toolchain "$TOOLCHAIN"
# 2. openai/math at the examined commit (download: repository) and its Mathlib cache (download: several GB).
git clone https://github.com/openai/math.git "$WORK/math"
git -C "$WORK/math" checkout --detach "$MATH_COMMIT"
cd "$WORK/math/lean"
test "$(cat lean-toolchain)" = "$TOOLCHAIN" || { echo "toolchain mismatch: $(cat lean-toolchain)"; exit 3; }
lake exe cache get
# 3. Comparator, landrun and lean4export from their official repositories; record the exact commits used.
for repo in leanprover/comparator Zouuup/landrun leanprover/lean4export; do
  name=$(basename "$repo"); git clone "https://github.com/$repo.git" "$WORK/$name"
  echo "$repo $(git -C "$WORK/$name" rev-parse HEAD)" >> "$OUT/run/tool_commits.txt"
done
# Build steps follow each repository's README at the recorded commit (to be pinned on first run):
#   comparator and lean4export: lake build, copy binaries to $WORK/bin; landrun: go build -o $WORK/bin/landrun
# 4. Record versions, configurations and allowed axioms, then run both comparisons in the documented isolation.
{ lean --version; lake --version; git -C "$WORK/math" rev-parse HEAD; uname -a; } > "$OUT/run/versions.txt"
cp ComparatorChallenges/DirectCrouzeix.json ComparatorChallenges/CourtadeKumar.json "$OUT/run/"
for challenge in DirectCrouzeix CourtadeKumar; do
  set +e
  systemd-run --property=RestrictAddressFamilies=~AF_UNIX --user --pty -E PATH="$PATH" --working-directory "$PWD" -- \
    bash -c "lake env comparator ComparatorChallenges/$challenge.json" > "$OUT/run/$challenge.stdout" 2> "$OUT/run/$challenge.stderr"
  echo $? > "$OUT/run/$challenge.exit_code"
  set -e
done
