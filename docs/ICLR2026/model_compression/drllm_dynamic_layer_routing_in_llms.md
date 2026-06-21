---
title: >-
  [论文解读] Dr.LLM: Dynamic Layer Routing in LLMs
description: >-
  [ICLR 2026][模型压缩][动态层路由] 给冻结的预训练 LLM 每层挂一个轻量路由器，让它决定该层「跳过 / 执行 / 重复」，用离线 MCTS 搜出的高质量路径做监督训练，在不改 base 权重、不做推理期搜索的前提下同时提升精度并节省计算。 领域现状：LLM 对每个 token 都固定地走完所有 transfo…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "动态层路由"
  - "自适应深度"
  - "层跳过/重复"
  - "MCTS 监督"
  - "冻结 LLM 改装"
---

# Dr.LLM: Dynamic Layer Routing in LLMs

**会议**: ICLR 2026  
**代码**: [https://github.com/parameterlab/dr-llm](https://github.com/parameterlab/dr-llm)  
**领域**: 模型压缩 / 自适应深度推理  
**关键词**: 动态层路由, 自适应深度, 层跳过/重复, MCTS 监督, 冻结 LLM 改装  

## 一句话总结
给冻结的预训练 LLM 每层挂一个轻量路由器，让它决定该层「跳过 / 执行 / 重复」，用离线 MCTS 搜出的高质量路径做监督训练，在不改 base 权重、不做推理期搜索的前提下同时提升精度并节省计算。

## 研究背景与动机
**领域现状**：LLM 对每个 token 都固定地走完所有 transformer 层，与输入难度无关。简单 query 上是计算浪费，难推理题上又缺乏「想得更深」的弹性。围绕自适应深度，已有早退出（early-exit）、层剪枝、循环/looped 块、动态路由、Mixture-of-Depth、MoE、搜索式路由（如 CoLa）等一大堆方案。

**现有痛点**：这些方法几乎都至少踩中三个坑之一——（i）用精度换速度，省了算力却掉点；（ii）要改架构 + 大规模重训（FlexiDepth、MindSkip 动辄几十万样本）；（iii）依赖推理期昂贵搜索，难以规模化部署（CoLa 在推理时还要 MCTS，且搜索时要 gold label 判断路径对错，实战不可用）。

**核心矛盾**：既想「改装到现成冻结模型上、训练廉价、推理零额外搜索、且 base 权重不动」，又想真正提精度而非掉点——现有工作没有一个能同时满足全部五条标准。

**本文目标**：做一个 retrofittable 框架，给冻结 LLM 配上每层的轻量路由器，做到 budget-aware、accuracy-driven 推理，且不动 base 权重、不做推理期搜索。

**核心 idea**：**显式监督的层路由**——把「该层跳/执行/重复」当成离线问题，用一种 length-aware MCTS 对每条样本搜出在算力预算下「保精度或涨精度」的执行路径，把这些路径转成每层标签，再用监督学习训练极小的路由器；推理时路由器直接贪心决策，彻底甩掉搜索。

## 方法详解

### 整体框架
给定 $L$ 层的冻结 decoder-only LLM $M=[B_1,\dots,B_L]$，Dr.LLM 给每个块 $B_\ell$ 配一个轻量 MLP 路由器 $r_\ell:\mathbb{R}^d\to\mathbb{R}^3$，输出 $\{\text{skip},\text{execute},\text{repeat}\}$ 三类 logits，决定该层是被绕过、执行一次、还是连续执行两次。训练分两段离线流程：先用 length-aware MCTS 为 ARC/DART 上每条样本搜出最优路径 $\pi^\star$ 并转成每层标签 $y^\star_\ell\in\{0,1,2\}$（仅 4k 条监督样本），再用 focal loss 把这些标签蒸进路由器，全程冻结 base 权重，只训路由器参数。推理时路由器按窗口池化后的隐状态贪心决策，零搜索、且兼容 KV cache。

```mermaid
flowchart TB
    subgraph 离线["离线监督生成 (MCTS)"]
        A[ARC/DART 样本 q,a] --> B[Length-aware MCTS<br/>预算内搜 skip/exec/repeat 路径]
        B --> C[保留保精度/涨精度路径 π*<br/>共 4k 条]
        C --> D[转每层标签 y*∈{0,1,2}]
    end
    subgraph 训练["路由器训练 (base 冻结)"]
        D --> E[Focal loss + 类别再平衡<br/>teacher forcing 仅执行]
        E --> F[每层路由器 r_ℓ]
    end
    subgraph 推理["推理 (无搜索)"]
        G[输入序列] --> H[窗口 mean-pool 隐状态]
        H --> F2[r_ℓ 贪心决策]
        F2 --> I[skip→透传 / exec→×1 / repeat→×2]
    end
    F --> F2
```

### 关键设计
**1. Skip/Execute/Repeat 三动作的每层路由器：把自适应深度变成离散分类。** 每层动作 $y_\ell\in\{\text{skip},\text{execute},\text{repeat}\}$，其中 skip 让 $H^{(\ell)}=H^{(\ell-1)}$ 直接透传、execute 应用 $B_\ell$ 一次、repeat 连续应用两次。整条向量 $y=(y_1,\dots,y_L)$ 诱导出一条定制执行路径，而 base 权重始终冻结。相比 LayerSkip 只能跳不能重复、且要重训 base，这里同时支持「省算力（skip）」和「按需加深（repeat）」两个方向，用 skip 抵消 looping 带来的层数增长，实现真正的算力再分配。路由器结构只是 Linear-GELU-Linear，宽度 $h=128$，每层只新增 $O(Ldh)$ 参数（3B 模型仅 11M、占 base 0.14%，8B 仅 16.8M、占 0.56%），单卡 A100 4 小时即可训完。

**2. 窗口 mean-pooling：在长上下文上稳住决策、且每序列只决策一次。** 路由器不读单 token，而是把前 $W\lfloor T/W\rfloor$ 个 token 切成 $W$ 个连续窗口 $\{S_w\}$，每窗取均值 $m_w=\frac{1}{|S_w|}\sum_{t\in S_w}H^{(\ell-1)}_t$，再对各窗 logits 取平均投票：$z_\ell=\frac{1}{W}\sum_w r_\ell(m_w),\ p_\ell=\mathrm{softmax}(z_\ell),\ \hat y_\ell=\arg\max_c p_{\ell,c}$（默认 $W=8$）。这样既抑制了长序列上逐 token 决策的抖动，又把路由器开销做成与生成 token 数无关的常数——每个输入序列只决策一次，推理 overhead 不到 1%，并与 KV cache 完全兼容（这是多数层路由方法做不到的）。

**3. Length-aware MCTS 离线造监督：把「该跳哪层」搜成训练标签。** 对每个 $(q,a)$，从默认路径 $\pi_0=[1,\dots,L]$ 出发，节点为「某层的某动作」，用带长度惩罚的 UCB 选择：$\mathrm{UCB}(\pi)=\frac{Q(\pi)}{v(\pi)}+c\sqrt{\frac{\ln V}{v(\pi)}}-\lambda\frac{|\pi|}{L}$（$c=1.8,\lambda=3.0$，10% 概率随机探索）。约束「连续 skip 至多两层、每块至多 repeat 一次、总路径长 $\le 2L$」控制算力膨胀；每次模拟跑一遍受约束前向，以二元 reward（答对/答错）回传，只保留相对 $\pi_0$「保精度或涨精度」的路径，并优先取最短的正确路径。仅 50 次模拟即可（CoLa 需 200 次），整个搜索 961k 次前向全在离线完成，最终收集 4k 条监督样本（约 30% 比默认路径精度更高），平均省 1.82 层。

**4. Focal loss + 类别再平衡 + 仅执行的 teacher forcing：对抗 execute 主导的极端类别不平衡。** 因为绝大多数层标签是 execute，直接训会让路由器永远预测「执行」。本文用 focal loss 配 effective-number 权重 $\alpha_c=\frac{1-\beta}{1-\beta^{n_c}}$（$\gamma=2,\beta=0.999$）：$L=-\frac{1}{L}\sum_\ell \alpha_{y^\star_\ell}(1-p_{\ell,y^\star_\ell})^\gamma\log p_{\ell,y^\star_\ell}$，把学习重心压到稀有的 skip/repeat 类上。训练时只对「执行」做 teacher forcing（用 $y^\star_\ell$ 替换路由器决策来沿标注路径走），从而让各层路由器互不依赖、不形成串行链——否则 $\text{router}_i$ 依赖 $\text{router}_{i-1}$ 输出会拖慢训练并掉 1.7% 精度。推理则纯贪心、无搜索。

## 实验关键数据
六个 backbone：LLaMA-3.2（3B/8B 的 Instruct/Base 共 4 个）、Qwen-2.5（3B/7B Instruct），路由器在单张 A100 40GB 上 4 小时训完、仅占 20% 显存。

### 主实验（in-domain：ARC 逻辑 + DART 数学）

| 指标 | 结果 |
|------|------|
| 精度提升（全部模型均涨） | 平均 +2.25%p，最高 +4.0%p |
| 单样本节省层数 | 平均约 5.0 层，最高 11.0 层 |
| 1k-token 生成提速 | 15.3%（路由器 overhead < 1%） |
| 对比 prior SoTA 路由方法 | 最高 +7.7%p 精度 |

### 消融 / 数据生成统计

| 设置 | 效果 |
|------|------|
| repeat 块大小 4→1 | 搜索大幅提速，精度/省层不变，模拟次数 200→50 |
| 长度惩罚 $\lambda$ 5→3 | 搜索样本减少 14.8%p |
| 去掉「仅执行 teacher forcing」（路由器串行依赖） | 精度掉 1.7%、训练变慢 |
| MCTS 总量 | 4k 监督样本 / 24,330 候选路径 / 961k 次前向（全离线） |

### 关键发现
- **OOD 泛化强**：路由器迁移到 MMLU、GSM8k、AIME、TruthfulQA、SQuADv2、GPQA、PIQA、AGIEval 等域外基准，平均仅掉 0.85%p 精度，同时保住效率——说明学到的是可迁移的路由策略而非过拟合特定任务。
- **三动作里 skip 与 repeat 互补**：repeat 在难推理题上加深、skip 在简单段省算力，整体实现「按难度重分配全局算力」，而不是单纯砍层掉点。
- **唯一同时满足五条标准**（涨精度 / 可改装 / 推理廉价 / 训练廉价 / base 不变），表 1 中其它方法均缺至少一项。

## 亮点与洞察
- **把「自适应深度」从架构问题降维成监督分类问题**：难点（哪层该跳）丢给离线 MCTS，推理期只剩一次廉价前向，干净地绕开了 CoLa 那种「推理期搜索 + 需要 gold label」的死穴。
- **retrofit 哲学贯彻彻底**：base 权重一动不动，只训 0.1~0.6% 的路由器参数，4 小时单卡搞定，部署友好且与 KV cache 兼容——这在层路由方法里相当罕见。
- **repeat 动作是点睛之笔**：多数自适应深度只做减法（早退/剪枝/跳层），Dr.LLM 允许对关键层加倍计算，并用 skip 抵消其算力开销，才能在省算力的同时反而涨精度。

## 局限与展望
- **监督只来自 ARC/DART 两类推理任务**：虽然 OOD 泛化不错，但 4k 样本、两个域的监督是否覆盖更广任务分布（如代码、长文档、多语种）仍待验证。
- **MCTS 离线成本不低**：961k 次前向虽是一次性离线开销，但换更大模型或更长上下文时搜索成本会上升，且 reward 依赖任务有明确 gold answer。
- **动作空间受限**：连续 skip ≤2、每层至多 repeat 一次、路径 ≤2L 等约束是为控算力膨胀而设的硬上限，可能限制极端难题需要的更深递归。
- **路由器按层独立决策**：靠 teacher forcing 解耦换来训练效率，但放弃了层间显式协同，未来可探索轻量的层间通信而不引入串行依赖。

## 相关工作与启发
- **vs. CoLa（Li et al. 2025）**：最接近的工作，同样把预训练层当模块、用 MCTS 搜「层的链」，但 CoLa 推理期还要搜索且依赖 gold label；Dr.LLM 把 MCTS 挪到离线、训练出路由器后推理零搜索。
- **vs. 早退出 / LayerSkip**：早退要附加分类器和校准，LayerSkip 要 finetune/重训 base 且只能跳不能重复；Dr.LLM 直接监督三动作、base 冻结。
- **vs. FlexiDepth / MindSkip**：同为改装路由，但要几十万样本且常掉点；Dr.LLM 仅 4k 样本且涨点。
- **vs. Mixture-of-Depth**：MoD 在 token 级路由但要改 base 权重，与本文 sequence 级 skip/exec/repeat 互补——token 级抓局部冗余，层级控全局算力，二者可结合。
- **启发**：「把昂贵的结构搜索离线化、再蒸成廉价 inference-time 策略」是一个通用范式，可推广到其它需要 input-adaptive 结构选择的场景（如自适应注意力跨度、动态专家选择）。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 「离线 MCTS 造监督 + 三动作层路由器 + base 冻结」的组合干净且切中部署痛点，repeat 动作和 length-aware 搜索是有辨识度的设计；不过单点技术（MCTS、focal loss、窗口池化）多为已知组件的巧妙拼装。
- **实验充分度**: ⭐⭐⭐⭐ — 六个 backbone 跨两家族、in-domain + 8 个 OOD 基准、对比多个 SoTA 路由方法、消融到搜索超参，证据链完整；可惜监督任务仅两类、未覆盖代码/长文等更广分布。
- **写作质量**: ⭐⭐⭐⭐ — 五条标准表格、动作定义、MCTS 算法与图示都清晰，方法可复现性强。
- **价值**: ⭐⭐⭐⭐ — retrofit 到冻结 LLM、单卡 4 小时、推理零搜索且涨精度省算力，工程落地价值高，对资源受限的自适应推理部署很实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Reassessing Layer Pruning in LLMs: New Insights and Methods](reassessing_layer_pruning_in_llms_new_insights_and_methods.md)
- [\[ICLR 2026\] Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM](alignment-enhanced_integration_of_connectivity_and_spectral_sparsity_in_dynamic_.md)
- [\[ICLR 2026\] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs](gradpruner_gradient-guided_layer_pruning_enabling_efficient_fine-tuning_and_infe.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](../../NeurIPS2025/model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)

</div>

<!-- RELATED:END -->
