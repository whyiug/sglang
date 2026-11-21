# Official v0.5.22, linux/amd64 (CUDA 13.0).
ARG SGLANG_BASE=lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e
FROM ${SGLANG_BASE} AS source
WORKDIR /opt/clef-auto
COPY LICENSE docker/clef/verify_runtime.py ./
COPY python/sglang/ python/sglang/

# Install the fork ahead of the base's editable SGLang package for all Python workers.
FROM source AS loader
RUN umask 022 && purelib="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" && \
    mkdir -p "/tmp/clef-loader$purelib" && \
    printf '%s\n' "import sys; sys.path.insert(0, '/opt/clef-auto/python')" \
      > "/tmp/clef-loader$purelib/clef_source.pth"

FROM source AS configured
COPY --from=loader /tmp/clef-loader/ /

ARG SGLANG_BASE
ARG FORK_COMMIT
ARG PYTHON_TREE
ARG UPSTREAM_COMMIT=75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7

FROM configured AS verified
RUN umask 022 && python3 -B verify_runtime.py \
    --base "$SGLANG_BASE" --fork "$FORK_COMMIT" --python-tree "$PYTHON_TREE" \
    --upstream "$UPSTREAM_COMMIT" > /tmp/clef-runtime.json

FROM configured
COPY --from=verified /tmp/clef-runtime.json /opt/clef-auto/runtime.json
ENV SGLANG_BUILD_COMMIT=${FORK_COMMIT}
LABEL org.opencontainers.image.title="SGLang CLEF automatic prefix cache" \
      org.opencontainers.image.base.name="${SGLANG_BASE}" \
      org.opencontainers.image.revision="${FORK_COMMIT}" \
      io.clef.sglang.upstream-commit="${UPSTREAM_COMMIT}" \
      io.clef.sglang.python-tree="${PYTHON_TREE}" \
      ai.sglang.build.commit="${FORK_COMMIT}"
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
