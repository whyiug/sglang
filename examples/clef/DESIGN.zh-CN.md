# CLEF 如何跨请求复用前缀

当两个请求送入同一个 CLEF 服务实例时，引擎可从相同的开头继续计算。调用方照常向 `POST /v1/systemone` 提交完整的 `state` 和本次的 `questions`。引擎按渲染和分词后的 token ID 匹配前缀，调用方无需标记固定文本、声明长度或在部署时登记指令。[请求协议](../../python/sglang/srt/entrypoints/systemone/protocol.py) · [快速使用](README.zh-CN.md)

## 哪些输入可以复用

联合头端点按模型约定把纯文本输入编码为「模板开头 + `state` 的 token + 本次问题及选项的 token + 模板结尾」。`questions` 可以包含一个或多个支持的问题，类型、ID、顺序和选项由本次请求给出。编码器为问题和模板预留空间，然后按剩余输入预算截取 `state`。最终送入模型的 token 序列决定能共享到哪里。[编码与输入预算](../../python/sglang/srt/entrypoints/systemone/joint_schema.py)

例如，两次请求的 `state` 都以同一段固定指令开始，后面各附一段不同的文本。两次分词结果从开头相同的部分都有机会复用，第一次出现不同 token 的位置即为匹配上限。如果两次 `state` 相同而问题发生变化，引擎也用同一条规则比较完整的输入。固定指令后实际相同的 token 可以一并匹配，中间发生分歧后再次出现的相同段落则需重新计算。[前缀树键与匹配](../../python/sglang/srt/mem_cache/common.py)

## 匹配了多长，本次就能跳过多长吗？

实际可跳过的长度取决于共同前缀内仍然有效的检查点。该模型的因果骨干同时包含全注意力和线性注意力层。要从共同前缀中某个位置继续计算，设备上必须同时具备相应的注意力 KV、卷积状态和 Gated DeltaNet 递推状态。调度器根据内核分块和检查点对齐要求选择可保存的位置，Triton 内核直接写出 FP32 递推快照。[检查点位置选择](../../python/sglang/srt/managers/schedule_batch.py) · [FP32 快照](../../python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py)

从空缓存串行发送带有共同开头、不同结尾的请求时，通常可以观察到以下过程。

| 请求 | 引擎行为 |
|---|---|
| 首次出现这段开头 | 计算完整输入，保存 token 路径和满足对齐条件的模型状态。 |
| 出现分叉 | 找到共同开头。若其中已有可恢复的检查点，本次就能复用。若需要新建分叉检查点，本次可能仍然全算，完成后发布该检查点。 |
| 再次经过同一分叉 | 若检查点已发布且仍驻留，引擎从该位置续算。并发请求的完成顺序、条目淘汰都会影响首次恢复的时机。 |

在仓库的一个 CPU 回归用例中，三个请求的前 213 个 token 相同，后缀各异。在该用例的 64-token 对齐配置下，第二笔匹配了 213 个 token，发布了位置 192 的检查点，第三笔实际跳过了前 192 个 token。[可运行的分叉与恢复用例](../../test/registered/unit/clef/test_radix_branch.py)

可以加请求头 `X-SGLang-Clef-Cache-Report: 1` 查看 `metadata.clef_prefix_cache`。`matched_tokens` 是本次请求进入调度时匹配到的 KV 长度，`cached_tokens` 是最终结果实际恢复的前缀长度，`published_tokens` 是本次新发布的检查点位置。用 `computed_tokens` 核对这次请求实际执行的 token 计算量，其中包含回退重试。刚发布的检查点可供后续请求使用。[调度计数](../../python/sglang/srt/managers/scheduler_components/output_streamer.py) · [HTTP 诊断](../../python/sglang/srt/entrypoints/systemone/serving.py)

## 命中后怎样给本次问题评分

CLEF 联合分类头需要看到完整序列的最终隐藏状态。每个驻留的物理 KV 槽额外保留一个 BF16 隐藏向量和一个 token ID。命中时，worker 按当前请求的槽位映射取出前缀，核对 token ID，再与本次新计算的后缀逐行拼接。分类头按当前请求中的问题位置和选项重新评分。因此，共享开头的请求可以有不同的问题和答案。保存这些向量会减少同样显存预算下可容纳的 token 数。[槽位恢复](../../python/sglang/srt/layers/clef_hidden_slots.py) · [联合分类头](../../python/sglang/srt/layers/joint_schema_head.py) · [显存估算](../../python/sglang/srt/model_executor/pool_configurator.py)

若命中了前缀却未得到有效评分，例如物理槽中记录的 token ID 与当前请求不符，端点丢弃该次结果，将同一输入与现有缓存隔离，重新完整计算一次。已取消的请求保留取消状态。维持正确性的其他条件包括缓存树原有的命名空间和 salt、检查点的发布时机，以及请求执行期间对相应槽位的持有。[缓存匹配](../../python/sglang/srt/mem_cache/common.py) · [检查点发布](../../python/sglang/srt/mem_cache/unified_radix_cache.py) · [回退与取消](../../python/sglang/srt/entrypoints/systemone/serving.py)

## 支持的执行配置

引擎在加载到 CLEF 联合头配置后检查稠密因果 Qwen3.5 骨干的实际结构，包括全注意力与线性注意力的层序、线性注意力的头数和头维度、卷积核宽度。Qwen3.5 的 0.8B、2B 和 4B 配置使用同一套判定。隐藏向量按模型宽度保存，分支检查点按该模型的递推状态形状保存。普通生成式检查点沿用 SGLang 原有推理路径。[能力判定](../../python/sglang/srt/clef_policy_cache.py) · [隐藏向量存储](../../python/sglang/srt/layers/clef_hidden_slots.py)

自动缓存要求 SGLang 原生模型实现、单张 NVIDIA GPU、固定权重、未量化 BF16、BF16 卷积状态和 FP32 递推状态。运行时采用确定性 Triton、设备本地 Python Radix 树、`extra_buffer` 检查点和单 token KV 页。新检查点可用下文的配对验收核实实际缓存命中和输出。[模型配置](../../python/sglang/srt/arg_groups/model_hook.py)

显式设置 `--disable-radix-cache` 可选择全量计算。LoRA、多卡、不兼容的注意力后端和外部缓存等设置均在这条优化路径的适用范围之外。单笔请求带有多模态输入、自定义输入嵌入或位置覆盖时，会跳过前缀读写。更换权重、tokenizer、模板或校准版本时应启动新实例。[请求级检查](../../python/sglang/srt/mem_cache/common.py) · [权重更新](../../python/sglang/srt/managers/scheduler_components/weight_updater.py)

## 查看差异与验证

在仓库根目录运行以下命令，可以查看本分支对已验收上游版本的引擎改动。

```bash
git diff --stat 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7 HEAD -- python/sglang
git diff 75a0d91c1f36a03e5213123fcdc4dbf0028ab7b7 HEAD -- python/sglang
```

可以在同一分支用 `--disable-radix-cache --enable-deterministic-inference --attention-backend triton --linear-attn-prefill-backend triton` 启动全量计算对照端。使用相同模型、tokenizer 和输入，确认自动缓存端有请求实际恢复了前缀，逐题核对完整公开答案，再独立比较经过预热的端到端时延和吞吐。若要将支持范围扩大到其他模型或执行配置，还需覆盖改变问题和前缀、检查点淘汰及失效重试。[部署与配对验收](README.zh-CN.md)
