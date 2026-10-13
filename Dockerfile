# syntax=docker/dockerfile:1
#
# Build (GIT_COMMIT is optional; .git isn't copied in, so this is how W&B runs record the commit):
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t go2-vision-rl .
#
# Run (host needs the NVIDIA Container Toolkit and driver >= 525, or >= 570 for Blackwell):
#   docker run --rm -it --gpus all --shm-size=8g -p 8080:8080 \
#     -e WANDB_API_KEY \
#     -v $PWD/logs:/workspace/logs -v $PWD/wandb:/workspace/wandb -v $PWD/onnx:/workspace/onnx \
#     -v go2-warp-cache:/root/.cache/warp \
#     go2-vision-rl train <TASK_ID> --env.scene.num-envs 4096 --agent.run-name <RUN_NAME>

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      tini \
      libegl1 \
      libgl1 \
      libgles2 \
      libglvnd0 \
      libglx0 \
      libopengl0 \
      libx11-6 \
      libxcursor1 \
      libxi6 \
      libxinerama1 \
      libxrandr2 \
      libxkbcommon0


ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

ARG PYTHON_VERSION=3.14.4
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_FROZEN=1 \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH="/opt/venv/bin:${PATH}"

RUN uv python install ${PYTHON_VERSION}

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

RUN ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg && \
    ffmpeg -hide_banner -version | head -n1

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY scripts ./scripts
RUN mkdir -p logs wandb onnx
ENV UV_NO_SYNC=1

ARG GIT_COMMIT=""
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
ENV WANDB_GIT_COMMIT="${GIT_COMMIT}"

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["list-envs"]
