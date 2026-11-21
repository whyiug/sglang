# 部署 CLEF 自动前缀缓存版 SGLang

本分支面向单张 NVIDIA GPU 上的 BF16 Qwen3.5-4B CLEF 联合分类模型。每次调用 `POST /v1/systemone` 时，`state` 传入待分析文本，`questions` 传入一个或多个问题，问题 ID 由调用方命名。

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

要观察跨请求复用，在本分支根目录运行下列标准库脚本。它生成三笔共享较长开头、后续文本各异的请求，对每笔报告输入 token 数、实际恢复量、新计算量以及客户端收到并解析全部答案的耗时。脚本通过可选请求头 `X-SGLang-Clef-Cache-Report: 1` 读取引擎计数。若第二笔才发现并建立分叉处的检查点，第三笔才会从该点恢复。问题数量和三笔请求的顺序是这个诊断示例的选择。[验收脚本](check_cache.py)

```bash
python3 examples/clef/check_cache.py --model clef --url http://127.0.0.1:38471
```

如需对照全算答案，可在**同一张卡分两次启动同一个镜像**，固定权重、模板、别名和其他参数。

1. 第一次在上面的 `docker run` 末尾追加 `--disable-radix-cache --enable-deterministic-inference --attention-backend triton`。服务就绪后保存本轮合成请求的确定性基准。
2. 停止该测试实例，用第二步原命令启动自动缓存服务，回放同一批请求。脚本会比较所有题目的完整 `answers`，包括置信度、选项概率与连续分数。退出码 0 表示本轮通过。

```bash
# 全算实例运行时执行
python3 examples/clef/check_cache.py --model clef --url http://127.0.0.1:38471 --save-baseline clef-reference.json
# 自动缓存实例运行时执行
python3 examples/clef/check_cache.py --model clef --url http://127.0.0.1:38471 --baseline-file clef-reference.json
```

有两台等配置实例时也可以运行 `python3 examples/clef/check_cache.py --model clef --url http://127.0.0.1:38471 --baseline-url http://127.0.0.1:38472`。三笔请求可核对恢复量、重算量与完整答案。测稳态吞吐和 P95 时应使用代表性负载，并记录并发度和预热条件。[正式矩阵与独立 100 笔对照](RESULTS.zh-CN.md)

## 4. 查看与维护改动

本分支从已做 GPU 验收的上游 `75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7` 建立。用以下命令可以查看引擎相对该版本的差异，或在安装好分支对应依赖的 Python 环境中运行 CPU 回归。容器常规启动选择服务入口，执行维护命令可用 `--entrypoint` 覆盖。

```bash
git diff --stat 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7...HEAD
git diff 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7...HEAD -- python/sglang
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/python" \
  python -B -m unittest discover -s test/registered/unit/clef -v
```

升级时记录上游提交、镜像基座摘要和成品镜像摘要。在目标驱动、GPU 和模型上完成全量计算与自动缓存的配对验收后，再替换服务镜像。回滚用前一份已验收的镜像摘要和对应模型版本。[设计和代码阅读路线](DESIGN.zh-CN.md) · [CPU 回归](../../test/registered/unit/clef)
