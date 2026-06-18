---
title: >-
  [论文解读] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
description: >-
  [ICML 2025][多模态VLM][剪枝] SparseVLM 提出了首个文本引导的免训练视觉 token 稀疏化框架，通过选择与视觉相关的文本 token 作为"评分者"来评估视觉 token 的重要性，结合自适应剪枝比率和 token 回收机制，在 LLaVA 上仅保留 192 个 token（减少 66.7%）时维持 99.1% 的原始性能。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "剪枝"
  - "VLM efficiency"
  - "text-guided sparsification"
  - "token recycling"
  - "training-free"
---

# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

**会议**: ICML 2025  
**arXiv**: [2410.04417](https://arxiv.org/abs/2410.04417)  
**代码**: [https://github.com/Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs)  
**领域**: 多模态VLM  
**关键词**: visual token pruning, VLM efficiency, text-guided sparsification, token recycling, training-free

## 一句话总结

SparseVLM 提出了首个文本引导的免训练视觉 token 稀疏化框架，通过选择与视觉相关的文本 token 作为"评分者"来评估视觉 token 的重要性，结合自适应剪枝比率和 token 回收机制，在 LLaVA 上仅保留 192 个 token（减少 66.7%）时维持 99.1% 的原始性能。

## 研究背景与动机

**领域现状**：当前视觉语言模型（VLMs）将高分辨率图像编码为大量视觉 token——一张 672×672 的图像在 LLaVA 中产生 2304 个视觉 token，占据超过一半的上下文长度。视频任务中 token 可达数千个，计算开销极大。

**现有痛点**：现有视觉 token 压缩方法存在两类问题：(1) 修改视像编码器或投影层的方法（如 Q-Former、DeCo）需要额外训练；(2) 在解码阶段剪枝的方法（如 FastV）忽略了文本 token 的引导作用，进行文本无关的稀疏化。面对不同问题时，模型可能需要关注图像的不同区域（前景或背景），固定的剪枝策略会丢失关键信息。

**核心矛盾**：图像信息本质上比文本稀疏（大量冗余），但现有方法要么需要额外训练开销，要么在剪枝时不考虑文本查询的语义引导，导致准确率和效率之间的 trade-off 过于激进。

**本文目标** (1) 如何在不需要训练的前提下实现高效的视觉 token 稀疏化；(2) 如何利用文本问题引导剪枝，根据具体查询自适应保留相关视觉信息；(3) 如何减少被剪枝 token 的信息损失。

**切入角度**：VLM 解码器的自注意力矩阵天然包含文本到视觉的交互信息，可以直接复用来判断视觉 token 的重要性，无需额外参数。

**核心 idea**：利用 VLM 已有的自注意力矩阵进行文本引导的视觉 token 筛选，通过矩阵秩自适应确定剪枝比率，并回收被剪 token 的信息来最小化损失。

## 方法详解

### 整体框架

SparseVLM 作为即插即用模块嵌入 VLM 解码器层。输入为图像和文本问题，先在进入解码器前预选"文本评分者"（与视觉相关的文本 token），然后在每个解码器层中提取自注意力矩阵的文本-视觉交互部分，计算视觉 token 的重要性分数，根据注意力矩阵的秩自适应确定本层的剪枝数量，最后将被剪 token 中信息较丰富的部分聚类压缩成紧凑的重建 token。

### 关键设计

1. **文本评分者选择（Text Rater Selection）**:

    - 功能：识别问题中与视觉内容真正相关的文本 token，排除介词、代词等无关词
    - 核心思路：在进入解码器前，计算视觉嵌入 $H_v$ 和文本嵌入 $H_q$ 之间的交叉注意力 $r = \frac{1}{L_v}\sum_{j=1}^{L_v}(\text{Softmax}(H_v H_q^T))_j$，选择相关性超过均值 $m=\text{mean}(r)$ 的文本 token 作为评分者。例如对药品相关问题，会选出"Tylenol"、"Advil"等关键词作为视觉筛选的依据
    - 设计动机：不是所有问题词都与视觉内容相关，让无关词参与评分会导致不准确的相关性计算。实验表明该选择机制在 POPE 上比不筛选提升 2.7%

2. **自适应稀疏化比率（Sparsification Level Adaptation）**:

    - 功能：根据每层的信息冗余程度，自适应确定该层需要剪枝的 token 数量
    - 核心思路：利用优先级矩阵 $P$（文本查询与视觉键的注意力子矩阵）的秩来衡量冗余度。维度与秩的差值表示冗余量，通过缩放因子 $\lambda$ 确定删除数 $N = \lambda \times (L_v - \text{rank}(P))$。矩阵秩高意味着视觉 token 之间线性独立程度高、冗余少，应少剪；秩低则冗余多，可多剪。若某层计算出 $N=0$，则跳过该层不剪枝
    - 设计动机：不同图像的信息密度不同，固定比率的剪枝策略在简单图像上浪费计算、在复杂图像上丢失信息

3. **视觉 Token 回收与重建（Token Recycling）**:

    - 功能：从被剪枝的 token 中回收信息较丰富的部分，压缩为紧凑的重建 token
    - 核心思路：从被删 token 中选取优先级最高的 $\tau$% 进行回收。使用密度峰值聚类（DPC）算法进行自适应聚类：先计算每个 token 的局部密度 $\rho_i$（基于 k 近邻距离），再计算距离指标 $\delta_i$（到更高密度 token 的最小距离），$\rho_i \times \delta_i$ 高的 token 成为聚类中心。最后将同一聚类内的 token 通过元素级求和压缩为单个重建 token
    - 设计动机：被剪 token 虽然重要性较低，但其中优先级相对较高的部分仍包含有用信息。回收机制尤其在高压缩率下效果显著—— POPE 上从 192 剪到 64 token 时，回收机制带来的提升从 1.5% 增至 17.7%

### 损失函数

SparseVLM 是免训练方法，不引入额外的损失函数。所有操作（注意力矩阵提取、评分者选择、秩计算、聚类回收）均在推理时在线完成。

## 实验关键数据

### 主实验表格

SparseLLaVA（LLaVA + SparseVLM）在8个基准上的表现（576→192 tokens，减少66.7%）：

| 方法 | GQA | MMB | MME | POPE | SQA | SEED | TextVQA | MMVet | 平均准确率 |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (576 tokens) | 61.9 | 64.6 | 1864 | 85.9 | 69.5 | 60.3 | 58.3 | 30.9 | 100% |
| FastV (192 tokens) | 52.6 | 61.0 | 1605 | 64.8 | 69.1 | 52.1 | 52.5 | 26.7 | 87.9% |
| SparseVLM (192 tokens) | 59.5 | 64.1 | 1787 | 85.3 | 68.7 | 58.7 | 57.8 | 33.1 | **99.1%** |

### 消融表格

Token 回收（TR）在不同压缩率下的效果（LLaVA 7B）：

| 基准 | 64 tokens | 96 tokens | 128 tokens | 192 tokens | 平均 |
|---|---|---|---|---|---|
| GQA | 52.2→53.8 | 55.2→56.4 | 58.1→58.4 | 59.4→59.5 | +0.8 |
| POPE | 72.8→77.5 | 77.5→81.9 | 83.7→85.0 | 85.2→85.3 | +2.6 |

### 关键发现

- 4.5× 视觉 token 压缩率下（576→128），SparseVLM 维持 96.7% 原始性能，仅下降 3.3%
- CUDA 延时减少 37%（57.82ms→36.50ms），FLOPs 减少 53.7%（4.62T→2.14T），同时准确率仅下降 0.9%
- 视频任务上（VideoLLaVA，90.5%剪枝率），SparseVLM 总平均准确率95.0%，比 FastV 高出 14.7 个百分点
- Qwen2-VL 上移除54.5%视觉 token 后维持 98.0% 准确率，验证了方法在动态分辨率模型上的通用性

## 亮点与洞察

- 该方法是首个免训练+文本引导的 VLM 视觉 token 稀疏化框架，即插即用无需微调，实用价值极高
- 密度峰值聚类的 token 回收策略设计精巧：不是简单丢弃被剪 token，而是从中回收信息并压缩，随着压缩率提高效果越显著
- 利用注意力矩阵的秩来 measure 冗余是一个巧妙的无参数化设计

## 局限性

- 需要从 FlashAttention 中提取完整的注意力矩阵，虽然论文提出了兼容方案但仍引入额外开销
- 矩阵秩的计算复杂度为 $O(L_t \times L_v \times \min(L_t, L_v))$，在序列很长时可能成为瓶颈
- 方法对超参数 $\lambda$、$\tau$、$\theta$ 的敏感性分析不够充分
- 仅在单图场景下验证；多图交错对话（multi-image interleaved dialogue）的适用性未探索
- 文本评分者选择在文本 token 极少时（如"Describe this image"）可能退化
- 尚未验证在最新的 LLaVA-OneVision、InternVL 2.5 等大规模 VLM 上的效果
- 逐层计算秩和聚类的开销在极长序列（如高分辨率视频）场景下可能成为瓶颈

### 补充分析

可视化结果（Figure 6）清晰展示了 SparseVLM 的注意力聚焦过程：随着解码层数增加，保留的 token 逐渐聚焦于与问题相关的区域（如问"what color"时聚焦于颜色区域），体现了文本引导的有效性。

## 相关工作与启发

- FastV（ECCV 2024）是最直接的对比方法，也在解码阶段剪枝但不考虑文本引导，SparseVLM 在所有设置下显著优于它
- ToMe（ICLR 2023）使用 token 合并策略但直接合并导致极端压缩下性能骤降
- 启发：文本引导的视觉处理思路可推广到其他多模态效率优化场景，如多模态检索、视频理解中的帧选择等
- PDrop（CVPR 2025）计算更轻量但在准确率和延时上均不如 SparseVLM，说明额外的注意力矩阵计算物有所值
- VocoLLaMA（Ye et al., 2025）需要训练的方案与 SparseVLM 免训练路线互补，未来可探索两者结合
- KV Cache 节省 67% 的特性对边缘设备和长上下文多模态推理场景有重要的工程应用价值

## 评分

⭐⭐⭐⭐ （7.5/10）

方法设计完整且实用——免训练、即插即用、跨模型通用、图像和视频都适用。在保持高精度的同时实现了显著的效率提升。实验充分（3个VLM架构、8个图像基准+4个视频基准），消融分析细致。稍显不足的是理论创新有限（各组件如矩阵秩、DPC聚类都是已有技术的组合），且超参数分析不够充分。

具体来说，SparseVLM 的实际部署价值很高：67% 的 KV Cache 内存节省（302.4MB→100.8MB）对于边缘设备推理意义重大。视频任务上 90.5% 剪枝率下仍保持 95% 准确率的表现也令人印象深刻。未来可探索的方向包括：与 KV Cache 压缩方法结合、针对多图多轮对话场景的优化、以及将文本引导思想扩展到音频-文本等其他模态组合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/multimodal_vlm/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)

</div>

<!-- RELATED:END -->
