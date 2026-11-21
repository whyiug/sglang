# CLEF 自动前缀缓存的工作原理

业务每次向 `POST /v1/systemone` 提交完整的 `state` 和本次的 `questions`。本文将 API 的 `state` 称为 **API state**，它由固定策略文本（policy 或 SP）与每笔不同的**业务 state** 拼成。常见输入是 1500-token policy、500-token 业务 state 和三道题。引擎从分词和模板渲染后的实际请求中发现共同的开头。部署端只需装载模型，新 policy 可以直接随请求发送。[快速使用](README.zh-CN.md)

## 为什么还需要缓存最终隐状态

在所选上游版本，CLEF 联合分类头走 embedding 模式，配置解析明确关闭 Radix Cache。因果 backbone 让相同 token 前缀产生相同表示，但这个 Qwen3.5-4B 同时有全注意力与线性注意力层。恢复前缀后，后续 token 计算需要该位置的注意力 KV、卷积状态以及 Gated DeltaNet 递推状态。随后，分类头要从完整序列中按本次题目的布局取数。普通的前缀 KV 命中只解决了其中一部分。[上游模型配置](https://github.com/sgl-project/sglang/blob/75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7/python/sglang/srt/arg_groups/model_hook.py) · [本分支分类头](../../python/sglang/srt/layers/joint_schema_head.py)

本分支在可恢复边界保存 FP32 递推检查点，把最终隐状态和 token ID 存在与设备 KV 槽对应的位置。worker 读取缓存前缀时核对 token ID，把已经恢复的前缀行和本笔新计算的后缀行按原序拼回。分类头仍使用本请求的题目、位置和完整表示重新评分。相同 policy 配不同业务 state、相同 API state 配不同题目，均按真实分词的公共开头处理。[FP32 检查点](../../python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py) · [按槽恢复](../../python/sglang/srt/layers/clef_hidden_slots.py)

## 一份新 policy 何时受益

| 从空缓存串行发送 | 引擎所做的事 |
|---|---|
| 第 1 笔，policy P + 业务 state A | 计算完整输入，按引擎对齐规则留下 token 路径和可恢复的检查点。 |
| 第 2 笔，同一 P + 业务 state B | 找到两笔共同开头。若共同部分内已有检查点，可以直接恢复。若只有更靠后的检查点，则在本笔计算并完成发布后，留下对齐到公共部分的分叉检查点。 |
| 第 3 笔，同一 P + 业务 state C | 若该分叉检查点仍驻留，恢复前缀，只计算后缀，再对全部题目评分。 |

队列查询与实际准入可能看到不同的树。诊断中的 `matched_tokens` 取实际准入时的 KV 匹配长度，`cached_tokens` 表示本次最终结果使用的恢复量，`published_tokens` 表示本次完成后新发布的检查点位置，`computed_tokens` 包含失效重试产生的计算。客户端加可选请求头 `X-SGLang-Clef-Cache-Report: 1` 后可以读取这些字段。普通请求照常只消费公开答案。并发首访、前序请求取消或条目淘汰都会影响首次有效命中的时机。[诊断字段来源](../../python/sglang/srt/managers/scheduler_components/output_streamer.py) · [三笔串行的实际计数](RESULTS.zh-CN.md)

## 代码阅读顺序

引擎迁移覆盖基础提交 `75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7` 之上的 21 个 Python 文件。以下命令从仓库根查看引擎差异，便于分别审查行为改动和打包文档。

```bash
git diff --stat 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7 HEAD -- python/sglang
git diff 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7 HEAD -- python/sglang
```

| 阅读位置 | 核对点 |
|---|---|
| [能力判定](../../python/sglang/srt/clef_policy_cache.py) · [模型配置](../../python/sglang/srt/arg_groups/model_hook.py) | 联合头、模型结构、设备和用户显式参数共同决定是否启用。HTTP、调度和 worker 读取同一份已解析的配置。 |
| [KV 匹配](../../python/sglang/srt/mem_cache/common.py) · [树后端](../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py) · [发布结果](../../python/sglang/srt/mem_cache/unified_radix_cache.py) · [调度](../../python/sglang/srt/managers/schedule_batch.py) | 使用引擎原有的 RadixKey、命名空间和 salt。仅在实际插入检查点并重新匹配后记录发布位置。隔离冷算请求跳过缓存读写。 |
| [Triton GDN](../../python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py) · [分块计算](../../python/sglang/kernels/ops/attention/fla/chunk.py) | 直接从计算中写出 FP32 快照。先舍入到 BF16 会影响恢复后各题的概率和分数。 |
| [容量配置](../../python/sglang/srt/model_executor/pool_configurator.py) · [worker 批信息](../../python/sglang/srt/model_executor/forward_batch_info.py) · [隐状态槽](../../python/sglang/srt/layers/clef_hidden_slots.py) · [联合头](../../python/sglang/srt/layers/joint_schema_head.py) | 为驻留槽预留最终隐状态与 token ID；写回新行、验证命中行，按当前请求还原完整序列。该额外显存会减少给定显存比例下的 token 池容量。 |
| [System One 端点](../../python/sglang/srt/entrypoints/systemone/serving.py) · [tokenizer manager](../../python/sglang/srt/managers/tokenizer_manager.py) · [输出传递](../../python/sglang/srt/managers/scheduler_components/output_streamer.py) | 若命中后得到空评分，则丢弃该结果，对同一输入仅尝试一次隔离的全量计算。计数同时包含两次尝试。已取消请求保留取消状态。 |
| [模型加载](../../python/sglang/srt/models/qwen3_5.py) · [激活内核](../../python/sglang/kernels/ops/activation/activation.py) · [换权重保护](../../python/sglang/srt/managers/scheduler_components/weight_updater.py) | 保留联合头 FP32 标量及每题型校准与置信度语义；另带可选 dense SiLU 预编译后端适配。活动缓存实例拒绝受保护的在线换权重路径。 |

## 适用范围与升级

已验证的组合为单 NVIDIA GPU、未量化 BF16、固定权重、纯文本、因果 Qwen3.5-4B 联合头、FP32 递推状态。自动路径使用确定性 Triton、设备本地 Python Radix 树、`extra_buffer` 检查点和单 token KV 页。显式关闭 Radix Cache、切换不兼容后端、LoRA、多卡、分离式或外部缓存等配置会触发相应的能力判定，原模型的其余部署限制仍然适用。多模态或自定义输入嵌入等请求在本实例内绕过前缀读写。模型权重、tokenizer、模板或校准有变化时启动新的实例。[配置判定代码](../../python/sglang/srt/clef_policy_cache.py) · [逐请求判定](../../python/sglang/srt/mem_cache/common.py)

移植到其他 SGLang 提交时，先审查上述接口、生命周期及引擎默认值，再用同模型的确定性全量计算端对照。至少覆盖换 policy、换业务 state、换题、淘汰及失效重试。逐题核对完整公开答案（包括置信度、选项概率和连续分数），再比较净计算量和端到端 P95。可用同一分支追加 `--disable-radix-cache --enable-deterministic-inference --attention-backend triton` 启动对照端。[构建与验收](README.zh-CN.md) · [已验收范围](RESULTS.zh-CN.md)
