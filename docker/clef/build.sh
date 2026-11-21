#!/usr/bin/env bash
# Build the Git revision recorded in the OCI labels. Only send committed runtime files.
set -euo pipefail

usage() {
  echo "Usage: bash docker/clef/build.sh [--check] [image:tag]"
}
if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
mode=(--load)
if [[ ${1:-} == --check ]]; then mode=(--check); shift; fi
if [[ $# -gt 1 || ${1:-} == -* ]]; then usage >&2; exit 2; fi

cd -- "$(dirname -- "$0")/../.."
inputs=(LICENSE python/sglang docker/clef/Dockerfile docker/clef/verify_runtime.py)
if [[ -n "$(git status --porcelain -- "${inputs[@]}" docker/clef/build.sh)" ]]; then
  echo "The release inputs have local changes. Commit them before building a revision-tagged image." >&2
  exit 1
fi
revision=$(git rev-parse HEAD)
tree=$(git rev-parse "$revision:python/sglang")
tag=${1:-local/clef-sglang:${revision:0:12}}
args=(--build-arg "FORK_COMMIT=$revision" --build-arg "PYTHON_TREE=$tree")
if [[ -n ${SGLANG_BASE:-} ]]; then
  if [[ ! $SGLANG_BASE =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]]; then
    echo "SGLANG_BASE must be an immutable image reference (registry/name@sha256:digest)." >&2
    exit 2
  fi
  args+=(--build-arg "SGLANG_BASE=$SGLANG_BASE")
fi

git -c tar.umask=0022 archive --format=tar "$revision" "${inputs[@]}" | \
  docker buildx build "${mode[@]}" --platform linux/amd64 --network=none \
    "${args[@]}" --file docker/clef/Dockerfile --tag "$tag" -
