---
title: >-
  [论文解读] Large Vision-Language Models Get Lost in Attention
description: >-
  [ICML 2026][多模态VLM][LVLM 可解释性] 本文用"信息复杂度 (eRank) + 子空间支持"的几何信息论框架定量诊断 LVLM 的残差流，发现 Attention 几乎只做子空间内重配置而 FFN 才注入新语义维度；更惊人的是把学习到的 attention 权重换成高斯噪声后多数视觉任务性能不降反升，揭示当代 LVLM 在 visual attention 上严重错配冗余。
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "LVLM 可解释性"
  - "注意力机制"
  - "信息论"
  - "子空间分析"
  - "注意力替换"
---

# Large Vision-Language Models Get Lost in Attention

**会议**: ICML 2026  
**arXiv**: [2605.05668](https://arxiv.org/abs/2605.05668)  
**代码**: 公开  
**领域**: 多模态 VLM / 可解释性  
**关键词**: LVLM 可解释性、Attention 冗余、信息论、子空间分析、注意力替换

## 一句话总结
本文用"信息复杂度 (eRank) + 子空间支持"的几何信息论框架定量诊断 LVLM 的残差流，发现 Attention 几乎只做子空间内重配置而 FFN 才注入新语义维度；更惊人的是把学习到的 attention 权重换成高斯噪声后多数视觉任务性能不降反升，揭示当代 LVLM 在 visual attention 上严重错配冗余。

## 研究背景与动机
**领域现状**：LVLM 的解码器仍是带残差连接的 Transformer，每个模块输出加性更新 $\Delta\mathbf{X}$ 写回共享残差流。主流假设是 attention 负责 in-context 推理（induction head、copy 机制），FFN 像 key-value memory 存事实。在 LVLM 里近期还发现 visual attention sink、visual attention drift 这些经验现象，提示模型可能没真正利用视觉证据。

**现有痛点**：现有分析大多停留在**统计层面**——画 attention 权重图、做 attention rollout、统计稀疏 head、做 causal intervention。但这些工具：(i) 缺乏统一的理论基础，难以横向比较不同模块、不同 metric 的结论；(ii) attention 权重本身已被 Jain & Wallace 等指出未必是可靠的归因信号；(iii) 没有量化"残差更新到底改变了表示什么"的统一度量。

**核心矛盾**：要回答"attention vs FFN 谁干了什么"必须有一个统一的、可对比的度量。LLM 表征分析社区已经用熵、有效秩等几何工具刻画跨层质量，但在 LVLM 模块级解释还是一片空白。

**本文目标**：(i) 定义"一个表示矩阵 $\mathbf{X}$ 蕴含什么信息"；(ii) 量化"加性更新 $\Delta\mathbf{X}$ 对 $\mathbf{X}$ 注入了什么"；(iii) 用这两个量诊断 LVLM 各模块的功能分工，特别是揭示 visual attention 是否真在做有意义的工作。

**切入角度**：把表示矩阵放到固定秩矩阵流形上看，用 SVD 自然得到"奇异谱 (复杂度)"和"列/行子空间 (语义支持)"两个几何对象；再用最小二乘的 innovation 概念量化更新里"超出现有子空间的能量"。这把模糊的"信息变化"问题变成可计算的子空间投影残差。

**核心 idea**：把 Transformer 残差更新分解为"创新 (RID) vs 重配置 (MixIG)"两个正交维度，再用这个透镜重新审视 LVLM。

## 方法详解

### 整体框架
针对残差流 $\mathbf{X}_{\text{new}} = \mathbf{X}_{\text{old}} + \Delta\mathbf{X}$，作者定义表示信息为 $\mathcal{I}(\mathbf{X}) = (\mathcal{S}_\mathbf{X}, \mathcal{D}_\mathbf{X})$：$\mathcal{S}_\mathbf{X} = \mathrm{eRank}(\mathbf{X})$ 描述谱复杂度，$\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$ 描述列/行子空间支持。然后用 **RID** 度量"外部创新"（谱变化 + 子空间新颖度），用 **MixIG** 度量"内部重配置"（token 级别混合熵的变化）。把这套度量分别套到 Attention 更新和 FFN 更新上，得到模块级功能分工的定量证据。

### 关键设计

**1. 用 SVD 几何刻画"一个表示矩阵里有什么信息"**

要回答 attention 和 FFN 谁干了什么，第一步得把"信息"说清楚——直接看 $\|\mathbf{X}\|_F$ 这种范数只看到能量大小，看不到结构。本文把表示矩阵放到固定秩流形 $\mathcal{M}_r = \{\mathbf{X} : \mathrm{rank}(\mathbf{X}) = r\}$ 上，用 SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ 拆出三类几何对象：左奇异子空间 $\mathcal{C}(\mathbf{X})$ 刻画 token 之间的关联、右奇异子空间 $\mathcal{R}(\mathbf{X})$ 刻画语义方向、奇异谱 $\mathbf{\Sigma}$ 刻画能量分布。在此基础上把复杂度归结为 effective rank $\mathcal{S}_\mathbf{X} = \exp(-\sum_i p_i \log p_i)$（$p_i = \sigma_i / \sum \sigma$），把语义支持归结为列/行子空间的投影算子对 $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$。这样一来，"改了多少有效维度"和"覆盖了哪些方向"被解耦开，后面才有可能区分一次更新到底是改了能量还是改了方向。

**2. RID：量化更新是否注入了"既不在原谱、也不在原子空间"的外部新信息**

有了表示信息的定义，就能问 $\Delta\mathbf{X}$ 究竟带来了什么。RID 把它拆成两块互补的量：谱变化 $\Delta\mathcal{S} = |\mathrm{eRank}(\mathbf{X}') - \mathrm{eRank}(\mathbf{X})| / \min(S, H)$ 抓"维度数变了没"，子空间创新 $\Delta\mathcal{D} = \frac{\|(\mathbf{I} - \mathbf{P}_{\mathcal{C}(\mathbf{X})})\mathbf{X}'\|_F + \|\mathbf{X}'(\mathbf{I} - \mathbf{P}_{\mathcal{R}(\mathbf{X})})\|_F}{2\|\mathbf{X}'\|_F}$ 借最小二乘的 innovation 概念抓"有多少能量落在原子空间之外"，合起来 $\mathrm{RID} = \Delta\mathcal{S} + \Delta\mathcal{D} \in [0, 2]$。两块缺一不可：只看谱变化会漏掉"换方向不换维度"的更新，只看子空间会漏掉"维度坍缩"。还有个务实的细节——RoPE 这类位置编码本身就让 RID 不为零，所以引入容差 $\epsilon_{\text{RoPE}} = \mathrm{RID}(\mathbf{X}^{(\text{RoPE})} \mid \mathbf{X}^{(\text{no-RoPE})})$ 当基线，把位置编码带来的"虚假创新"扣掉，避免误判。

**3. MixIG + 噪声替换：量化"子空间内的 token 重排"，并把度量挂到真实性能上**

RID 看不见的是另一类更新——不引入新方向，只在已有子空间里把 token 重新搅匀。MixIG 补这个洞：把每行 token 归一化后构造 token-to-token 混合分布 $P_{t,j} \propto \frac{\tilde{\mathbf{x}}_t^\top \tilde{\mathbf{x}}_j + 1}{2}$，取平均 Shannon 熵得 TME，$\mathrm{MixIG} = \mathrm{TME}(\mathbf{X}') - \mathrm{TME}(\mathbf{X})$，正值意味着更新让 token 互相混合得更广。但光有几何度量还不够说服力，作者再设计了一个受控替换实验把它和下游表现挂钩：在 15 个开源 LVLM 上把 attention 更新换成两种噪声——Noise $\mathbf{\Delta}$ 直接用高斯噪声替掉 $\Delta\mathbf{X}_{\text{attn}}$，Noise $\mathbf{QKV}$ 用高斯权重替掉 Q/K/V 矩阵——看性能和几何信号怎么动。逻辑很直接：如果 attention 真的在做有意义的工作，把它换成随机就应该崩；结果多数视觉任务不降反升，正好印证了 MixIG/RID 给出的"attention 几乎只做子空间内重排、不注入新信息"的判断。

### 损失函数 / 训练策略
本文是诊断框架，不训练新模型。所有度量都是 forward-pass 上的几何量。实验在 Qwen2.5-VL / LLaVA-1.5 / LLaVA-NeXT 三个家族共 15 个变体上跑 POPE、3DSRBench、RealWorldQA、MMMU、VMCBench、MathVista、HallusionBench 等 7 个 benchmark，每类 1000 个样本统计。

## 实验关键数据

### 主实验
跨模型聚合的模块级 RID/MixIG (论文 Table 1)：

| 模块 | RID | MixIG | 功能特征 |
|------|-----|-------|---------|
| Noise $\mathbf{\Delta}$ | 0.61 | -0.80 | 高 RID + 负 MixIG (off-manifold 扰动) |
| Noise $\mathbf{QKV}$ | 0.44 | -0.50 | 高 RID + 负 MixIG |
| **Attention** | **0.06** | **0.61** | **低 RID + 高 MixIG** (子空间保持 + 重配置) |
| **FFN** | **0.21** | **0.02** | **高 RID + 低 MixIG** (子空间扩张 + 创新) |

横跨 15 个模型的分离非常稳定：attention 的 RID 几乎等于 $\epsilon_{\text{RoPE}} = 0.062$，说明它**几乎不引入新支持方向**，全是混合；FFN 的 RID 显著高于此基线，是真正的创新源。

### 消融实验
SAP (Stochastic Attention Probing) 噪声替换实验（部分摘自 Table 2，Qwen-2.5-VL-3B）：

| 配置 | POPE | RWQA | 3dSRBench |
|------|------|------|-----------|
| Vanilla | 86.13 | 59.35 | 53.46 |
| + Vis. Attn. (噪声替换) | 87.58 | 61.38 | — |

在多数视觉任务上，**用高斯噪声替换学习到的 visual attention 权重，性能不降反升**——这是本文最戏剧性的发现。

### 关键发现
- Attention 与 FFN 的功能在几何上正交：Attention = 子空间保持算子（reconfiguration）；FFN = 子空间扩张算子（innovation）。前人的"attention 做 in-context、FFN 做记忆"假设被几何证据夯实。
- LVLM 的 visual attention 大量冗余，attention scores 携带的有效信息很少；这与 attention sink、attention drift 现象互为佐证。
- 既然 attention 复杂度是 $O(S^2)$ 的主要瓶颈但又冗余，本文实质上为视觉 token 上的近似 attention（稀疏/预定义/低秩）提供了强有力的理论与经验依据。

## 亮点与洞察
- "RID + MixIG" 是一组优雅的对偶度量，把残差更新分解成"加入新基底"与"在旧基底里搅匀"。这套语言比 attention rollout、tuned lens 这种工具级别的方法更通用，可以套到任何加性更新模块上。
- 噪声替换实验比任何 ablation 都激进——把"学到的权重换成完全随机"竟然不影响视觉任务，意味着 LVLM 训练目标可能对 visual attention 的学习信号过弱。这给"视觉 token pruning、attention free 视觉融合"等方向提供了直接动机。
- RoPE 基线的引入很务实——把位置编码引入的"虚假 RID"扣掉，避免误判，体现作者把度量做成可移植工具的工程意识。

## 局限与展望
- 框架只看单步加性更新，跨层累积效应（哪些层连起来才是真创新）需要更多分析。
- 噪声替换实验集中在视觉端，对纯文本任务（MathVista）的 attention 仍很关键，本文没解释为什么文本-视觉的依赖是非对称的。
- 度量都是相对量，缺一个"绝对信息量"基准；不同模型的 RID 数值跨模型可比性还需要更多控制实验。

## 相关工作与启发
- **vs Tuned Lens / 线性探针**：探针只能告诉你某层"含什么"，本文告诉你某模块"加了什么"，颗粒度更细。
- **vs Attention Sink / Drift 经验研究**：那些工作发现现象，本文给现象提供几何信息论解释——sink 本质就是 attention 在 reconfiguration 时把熵集中到少数 token。
- **vs Sparse Attention / Attention-Free 模型**：本文等于为这些工作提供了理论后台——既然 attention scores 冗余，那把它换成线性或固定 pattern 不损失视觉能力是合理的。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把信息几何首次系统化引入 LVLM 模块解释，并给出反直觉的噪声替换发现。
- 实验充分度: ⭐⭐⭐⭐ 15 个模型 × 7 个 benchmark，覆盖面广。
- 写作质量: ⭐⭐⭐⭐ 三个 RQ 渐进式推进，定义-度量-诊断逻辑清晰。
- 价值: ⭐⭐⭐⭐⭐ 直接动摇 LVLM 视觉路径的设计假设，对架构和效率研究都有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)

</div>

<!-- RELATED:END -->
