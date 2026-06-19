---
title: >-
  [论文解读] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding
description: >-
  [CVPR 2025][视频理解][动态编码] 提出DynFocus，一个基于LLM的动态协作视频编码网络，通过DPE模块动态选择与问答相关的关键帧，CCE模块对关键帧用细粒度token编码（类似视锥细胞Cones）、对冗余帧用极少token粗粒度编码（类似视杆细胞Rods），在有限token预算下平衡空间细节与时序动态。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "动态编码"
  - "内存高效"
  - "大语言模型"
  - "token压缩"
---

# DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2411.12355](https://arxiv.org/abs/2411.12355)  
**代码**: [https://github.com/Simon98-AI/DynFocus](https://github.com/Simon98-AI/DynFocus)  
**领域**: 视频理解  
**关键词**: 视频理解, 动态编码, 内存高效, 大语言模型, token压缩

## 一句话总结

提出DynFocus，一个基于LLM的动态协作视频编码网络，通过DPE模块动态选择与问答相关的关键帧，CCE模块对关键帧用细粒度token编码（类似视锥细胞Cones）、对冗余帧用极少token粗粒度编码（类似视杆细胞Rods），在有限token预算下平衡空间细节与时序动态。

## 研究背景与动机

LLM-based视频理解的核心矛盾是：**长视频需要大量token才能保留视觉语义信息，但LLM的内存/上下文长度有限**。现有方法的不足：

1. **空间压缩方法**（如average pooling、attention、dynamic masking）：丢弃关键视觉细节
2. **时间采样方法**（如均匀采样部分帧）：可能遗漏关键帧
3. **记忆库方法**（如MovieChat、MA-LMM）：关键帧随问题变化，固定记忆库无法灵活适配

作者通过统计学习发现两个重要观察：
- **冗余性**：视频中大量帧是重复的或与答案无关的（如ActivityNet中约60-70%帧是冗余的）
- **对应性**：不同问题需要关注视频中不同的帧，即"关键帧"是问题相关的

这启发了动态编码策略：根据帧与问题的相关性，分配不同数量的token。

## 方法详解

### 整体框架

DynFocus分三部分：(1) 视觉+文本编码器提取特征；(2) 动态协作网络（DPE + CCE）作为connector压缩视频token；(3) LLM接收压缩token生成回答。

### 关键设计

1. **动态事件原型估计模块 (DPE - Dynamic Event Prototype Estimation)**:
    - 功能：从T帧视频中动态选择与问答最相关的K个关键帧
    - 核心思路：分两步去除冗余。**第一步（去时间冗余）**：对每帧先做空间平均池化（$N \rightarrow P$ patches），然后用DPC-KNN聚类在时间维度上获得L个事件原型。聚类使用局部密度 $\rho_t = \exp(-\frac{1}{C}\sum_{t' \in \mathcal{N}(t)} \frac{1}{P}\|\mathbf{f}_t - \mathbf{f}_{t'}\|_F^2)$ 和距离指标 $\delta_t$，取 $\rho_t \times \delta_t$ 最高的L个。**第二步（去答案无关冗余）**：用MLP网络 $\mathcal{U}(\cdot)$ 回归帧级打分 $s_l = \mathcal{U}(\text{Max}(\mathbf{m}_l) || \text{Avg}(\mathbf{m}_l))$，取Top-K个最高分原型
    - 设计动机：DPC-KNN聚类是无参数的，能自适应发现视频中的"事件"；Top-K操作不可微，使用perturbed maximum方法转为线性规划问题实现端到端训练：$\mathbf{P}_\sigma = \mathbb{E}_P[\text{argmax}_{\mathbf{P} \in \mathcal{C}} \langle \mathbf{P}, \mathbf{s}\mathbf{1}^\top + \sigma \mathbf{Z} \rangle]$，使得打分网络能通过LLM的autoregressive loss梯度更新

2. **紧凑协作编码模块 (CCE - Compact Cooperative Encoding)**:
    - 功能：对关键帧和冗余帧分别编码，前者保留细节，后者保留概要
    - 核心思路：灵感来自灵长类视网膜细胞 — **Cones编码**（对$b_t=1$的关键帧）：串联事件原型和多粒度空间物体原型（通过多层空间DPC-KNN聚类获得），用MLP映射 $\mathbf{U}_{t,b_t=1} = \mathcal{F}_{fine}(\mathbf{h}_t || \mathbf{G}_t)$，保留全部空间token；**Rods编码**（对$b_t=0$的冗余帧）：用文本embedding $\mathbf{Q}$ 做cross-attention调制 $\mathbf{E} = \text{Softmax}(\frac{f_q(\mathbf{G}_t) f_k(\mathbf{Q})^\top}{\sqrt{d}}) \mathbf{G}_t$，然后average pooling压缩为仅2个token $\mathbf{U}_{t,b_t=0} = \mathcal{F}_{coarse}(\text{Avg}(\mathbf{E}) || \text{Avg}(\mathbf{G}_t))$
    - 设计动机：关键帧需要精细的空间细节（如识别物体属性），冗余帧只需提供时间线索（如事件顺序）；2个token足以保持帧间时序连贯性，同时大幅压缩token量

3. **协作编码融合**:
    - 功能：将Cones和Rods编码统一
    - 核心思路：$\mathbf{O}_t = b_t \cdot (\mathbf{U}_{t,b_t=1} || \mathbf{U}_{t,b_t=0}) + (1-b_t) \cdot \mathbf{U}_{t,b_t=0}$，关键帧同时保留细粒度和粗粒度信息，冗余帧仅保留粗粒度
    - 设计动机：关键帧的Rods编码也被保留，提供该帧的时序上下文；这种cooperative设计确保了信息互补

### 损失函数 / 训练策略

**两阶段训练**：
- **Stage 1（视觉-语言对齐）**：冻结视觉编码器+LLM，只训练动态协作网络的投影层。使用LLaVA-filter-CC3M图像caption + WebVid-2.5M视频caption
- **Stage 2（指令微调）**：全量微调LLM + DPE + CCE全部参数。使用LLaVA-665K图像QA + ScienceQA + VideoChat2子集（VideoChatGPT-100K + NExT-QA + CLEVRER等）

使用ViT-G/14 (EVA-CLIP)作为视觉编码器，InstructBLIP的Qformer作为文本编码器，Vicuna-7B-1.5作为LLM。8×A100 80G训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DynFocus | 之前SOTA (7B) | 提升 |
|--------|------|------|----------|------|
| MSVD-QA | Acc/Score | 74.8/4.0 | ST-LLM 74.6/3.9 | 持平 |
| MSRVTT-QA | Acc/Score | 62.8/3.6 | ST-LLM 63.2/3.4 | Score更高 |
| ANet-QA | Acc/Score | 50.3/3.4 | ST-LLM 50.9/3.3 | 用更少token |
| MLVU M-Avg | Multi-choice | 49.6% | MiniGPT4-V 44.5% | +5.1% |
| MLVU G-Avg | Generation | 4.38 | MiniGPT4-V 3.36 | +1.02 |
| LV-Bench | Overall | 32.9% | LLaVA-NeXT-34B 32.2% | 7B超34B |
| VideoMME Overall (w/o subs) | Accuracy | 44.1% | ST-LLM 37.9% | +6.2% |

### 消融实验

| 配置 ($|\mathbf{U}_{b_t=0}|$ / $|\mathbf{U}_{b_t=1}|$) | MSVD-QA Acc | ANet-QA Acc | 说明 |
|------|---------|------|------|
| 0 / 40 (无Rods) | 63.7% | 41.4% | 丢失时序信息 |
| 2 / 0 (无Cones) | 58.2% | 38.6% | 丢失空间细节，严重下降 |
| 2 / 2 (均粗编码) | 62.0% | 40.5% | 关键帧也压缩则性能差 |
| 2 / 256 (不压缩Cones) | 68.4% | 44.3% | token最多，最佳但内存大 |
| **2 / 40 (本文默认)** | **67.9%** | **43.1%** | 性能接近不压缩，token大幅减少 |

### 关键发现

- 初始事件原型数L=25为最佳：太少（<10）无法覆盖关键事件，太多（>25）破坏时序结构
- 筛选比例K/L=0.8为短视频最佳；长视频（如LV-Bench）需要更大的L和更小的K/L来处理更多内容
- 去掉Cones编码（仅用Rods）导致MSVD-QA下降9.7%，说明空间细节至关重要
- 去掉Rods编码（仅用Cones）也下降4.2%，说明冗余帧的时序信息不可忽视
- DynFocus无字幕版本（44.1%）超越ST-LLM有字幕版本（42.3%），说明动态编码能有效弥补信息缺失

## 亮点与洞察

- **生物学启发独特且合理**：Cones/Rods的类比不仅是修辞，实际对应了精细/粗粒度编码的功能分工，设计哲学有说服力
- **端到端可微的动态选择**：通过perturbed maximum方法解决Top-K不可微问题，使DPE打分网络能被LLM的loss隐式监督
- **token效率极高**：在MLVU上DynFocus用的token数远少于竞争方法，但性能最优
- 框架通用，视觉编码器可替换为其他clip-based编码器

## 局限与展望

- 对ego-centric视频（如ER任务）效果较弱，需要针对性的ego视频数据
- DPC-KNN聚类的超参数（近邻数C等）需要手动设置
- 仅在7B模型上验证，更大规模LLM上的效果未知
- 可探索将文本引导直接融入DPE的帧选择过程（当前DPE仅使用视觉特征选帧）

## 相关工作与启发

- 与LLaMA-VID的关系：LLaMA-VID用dual-token（context+content）表示每帧，DynFocus则动态分配不同粒度
- 与Chat-UniVi的关系：Chat-UniVi用token merging压缩，DynFocus用聚类+重要性打分更灵活
- 启发：动态编码思想可推广到其他长序列模态（如长音频、长文档的chunk-level动态表示）

## 评分
- 新颖性: ⭐⭐⭐⭐ DPE端到端可微选帧+CCE Cones/Rods协作编码，思路清晰且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5个benchmark覆盖短视频/长视频/幻觉检测，消融全面
- 写作质量: ⭐⭐⭐⭐ 生物学类比恰当，公式严谨
- 价值: ⭐⭐⭐⭐ 为LLM视频理解的token效率问题提供了有效解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2025\] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
