# 部署 CLEF 自动前缀缓存版 SGLang

基于 Qwen3.5 稠密骨干的 0.8B、2B 和 4B CLEF 联合分类模型共用一个镜像和启动命令。把 `CLEF_MODEL` 指向相应的完整 CLEF 检查点，服务在单张 NVIDIA GPU 上读取模型配置，为满足条件的 BF16 模型自动启用前缀缓存。每次调用 `POST /v1/systemone` 时，`state` 传入待分析文本，`questions` 传入一个或多个问题。

如果几次请求的文本以相同的固定指令开头，服务会按模板渲染、分词后的 token 序列从头匹配，在已驻留且可恢复的检查点继续计算。普通文本的共同开头也能复用。调用方无需标注前缀、设置长度或约定分隔符。每次请求都可以更换文本、问题或两者。更换权重、tokenizer 或模板时启动新实例。[工作原理与模型适用范围](DESIGN.zh-CN.md)

## 1. 在构建机制作镜像

需要 Git 和支持 Buildx 的 Docker。以下命令在 Linux/amd64 构建机上运行，无需模型或 GPU。首次读取官方基座约需下载 15.6 GB 压缩层。脚本从所选 Git 提交导出运行所需的文件，镜像内会校验 CUDA、关键 Python 包、JIT 编译工具以及实际加载的引擎路径。

```bash
git clone --single-branch --branch clef/auto-prefix-cache https://github.com/whyiug/sglang.git
cd sglang
IMAGE="local/clef-sglang:$(git rev-parse --short=12 HEAD)"
bash docker/clef/build.sh "$IMAGE"
docker run --rm --entrypoint cat "$IMAGE" /opt/clef-auto/runtime.json
```

仅检查 Dockerfile 与基座元数据可以运行 `bash docker/clef/build.sh --check`。构建默认从固定摘要的 SGLang 官方 Linux/amd64 镜像取二进制环境，镜像内的 Python 引擎来自这个 fork。查看 `runtime.json` 中的 `fork_commit`、`python_tree`、`base` 和 `abi_packages` 可追溯实际构建输入。[镜像配方](../../docker/clef/Dockerfile) · [运行环境校验](../../docker/clef/verify_runtime.py)

## 2. 在推理机启动服务

在构建机给成品镜像加上团队镜像仓库的标签并推送。推理机可直接用推送返回的 digest 部署，也可用 `docker save` / `docker load` 离线交付。推理机需要兼容镜像 CUDA 的 NVIDIA 驱动、NVIDIA Container Toolkit，以及本地完整 CLEF 模型目录（合并后的权重、联合头、tokenizer、模板和模型校准文件）。已有权重可直接只读挂载。

```bash
# 构建机，把 REGISTRY_IMAGE 设为你有写权限的仓库，例如 registry.example.com/team/clef-sglang
: "${REGISTRY_IMAGE:?请设置团队镜像仓库}"
TAG=$(git rev-parse --short=12 HEAD)
docker tag "$IMAGE" "$REGISTRY_IMAGE:$TAG"
docker push "$REGISTRY_IMAGE:$TAG"
```

推理机将 `IMAGE` 设为团队仓库的镜像引用，例如 `registry.example.com/team/clef-sglang@sha256:…`。本机试用可沿用第一步的 `IMAGE`。示例使用模型别名 `clef`、GPU 0 和端口 38471，显存静态分配比例设为 0.5。远端客户端可通过内部网关连接该端口。

```bash
: "${IMAGE:?请设置派生镜像}" "${CLEF_MODEL:?请设置本地 CLEF 模型目录}"
CLEF_GPU=${CLEF_GPU:-0}
PORT=${PORT:-38471}
docker run --rm --name clef-auto --gpus "device=$CLEF_GPU" --shm-size 1g \
  -p "127.0.0.1:$PORT:30000" \
  --mount "type=bind,src=$(realpath "$CLEF_MODEL"),dst=/models/clef,readonly" \
  "$IMAGE" --model-path /models/clef --served-model-name clef \
  --host 0.0.0.0 --port 30000 --context-length 8192 --mem-fraction-static 0.5
```

## 3. 发出请求并验证缓存

服务就绪后，可先用一个问题验证接口。`ordered` 是调用方为这道题取的名字，`noul` 是 CLEF 接口中是非题的类型名。响应里的 `answers.ordered` 对应这道题。增加或更换问题时，在 `questions` 中提供相应条目即可。[请求和响应字段](../../python/sglang/srt/entrypoints/systemone/protocol.py)

```bash
curl --fail-with-body -sS http://127.0.0.1:38471/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef","state":"字母序列为 A、B、C。","questions":{"ordered":{"type":"noul","instructions":"字母是否按字母表顺序排列？"}}}'
```

要观察跨请求复用，可以给普通请求加上诊断头 `X-SGLang-Clef-Cache-Report: 1`。下面用 Python 标准库发送三个共享开头、结尾各异的请求。每次运行都生成新的文档编号，方便在已经启动的实例上观察新前缀的发现过程。

```bash
python3 - <<'PY'
import json
from urllib.request import Request, urlopen
from uuid import uuid4

url = "http://127.0.0.1:38471/v1/systemone"
prefix = f"Document {uuid4().hex}\n" + "Read the letters in their original order.\n" * 80
questions = {"ordered": {"type": "noul", "instructions": "Are the letters in alphabetical order?"}}
for suffix in ("A B C", "C B A", "B A C"):
    payload = {"model": "clef", "state": prefix + suffix, "questions": questions}
    headers = {"Content-Type": "application/json", "X-SGLang-Clef-Cache-Report": "1"}
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    print(suffix, (result.get("metadata") or {}).get("clef_prefix_cache"))
PY
```

`matched_tokens` 是匹配到的长度，`cached_tokens` 是本次实际恢复的长度，`computed_tokens` 是本次实际计算量。若第二个请求才建立分叉检查点，后续请求可从该点恢复。输出为 `None` 时，可查看启动日志中记录的模型和运行配置是否启用了自动缓存。[缓存诊断与适用范围](DESIGN.zh-CN.md)

更换检查点后，可分两次启动同一镜像验收。全量计算实例在启动命令末尾追加 `--disable-radix-cache --enable-deterministic-inference --attention-backend triton --linear-attn-prefill-backend triton`，自动缓存实例使用上面的启动命令。向两端回放同一批共享前缀、后缀不同的请求，按问题 ID 逐项核对完整的 `answers` 字段，同时确认自动缓存实例有请求返回 `cached_tokens > 0`。这样能实际覆盖从前缀检查点恢复的计算路径。

## 4. 查看与维护改动

本分支从已做 GPU 验收的上游 `75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7` 建立。用以下命令可以查看引擎相对该版本的差异，或在安装好分支对应依赖的 Python 环境中运行 CPU 回归。容器常规启动选择服务入口，执行维护命令可用 `--entrypoint` 覆盖。

```bash
git diff --stat 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7...HEAD
git diff 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7...HEAD -- python/sglang
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/python" \
  python -B -m unittest discover -s test/registered/unit/clef -v
```

升级时记录上游提交、镜像基座摘要和成品镜像摘要。在目标驱动、GPU 和模型上完成全量计算与自动缓存的配对验收后，再替换服务镜像。回滚用前一份已验收的镜像摘要和对应模型版本。[设计和代码阅读路线](DESIGN.zh-CN.md) · [CPU 回归](../../test/registered/unit/clef)
