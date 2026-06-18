---
title: >-
  [论文解读] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models
description: >-
  [CVPR2026][多模态VLM][扩散语言模型] 首次定量分析扩散多模态LLM (dMLLM)的CoT推理过程，发现"早期回答生成"和"弱视觉依赖"两个关键问题，提出PSP（位置-步骤惩罚）和VRG（视觉推理引导）两种免训练方法，在3倍加速下获得最高7.5%的精度提升。 领域现状：扩散LLM (dLLM)如LLaDA、D…
tags:
  - "CVPR2026"
  - "多模态VLM"
  - "扩散语言模型"
  - "多模态推理"
  - "Chain-of-Thought"
  - "视觉引导"
  - "重掩码策略"
---

# Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models

**会议**: CVPR2026  
**arXiv**: [2604.05497](https://arxiv.org/abs/2604.05497)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 扩散语言模型, 多模态推理, Chain-of-Thought, 视觉引导, 重掩码策略

## 一句话总结
首次定量分析扩散多模态LLM (dMLLM)的CoT推理过程，发现"早期回答生成"和"弱视觉依赖"两个关键问题，提出PSP（位置-步骤惩罚）和VRG（视觉推理引导）两种免训练方法，在3倍加速下获得最高7.5%的精度提升。

## 研究背景与动机

**领域现状**：扩散LLM (dLLM)如LLaDA、Dream是自回归LLM的新兴替代方案，通过并行恢复多个token提供更快推理，扩展到多模态便形成dMLLM。然而dMLLM的推理过程尚未被充分理解，作者做了两个关键观察：

1. **Early Answer Generation**：dMLLM在很早的时间步就生成最终答案token（L=64/T=32时，30%+在第7步前就确定答案），然后才生成中间推理来合理化答案
2. **Weak Visual Grounding**：初始时间步中dMLLM对视觉prompt的依赖极低（PDM值低），这与AR-VLM形成鲜明对比——AR模型在早期高度依赖视觉特征

结论：dMLLM倾向于在充分利用视觉输入之前就过早生成答案。

## 方法详解

### 整体框架

针对上面两个毛病——过早定答案、初期不看图——作者提出两个完全免训练的推理时方法：位置-步骤惩罚 (Position & Step Penalty, PSP) 在早期时间步压住"急着生成答案"的冲动，视觉推理引导 (Visual Reasoning Guidance, VRG) 在生成时放大视觉条件信号。两者都嵌入 dMLLM 的扩散去噪循环：每个时间步里，PSP 修正 token 置信度、VRG 修正 logits，再共同决定这一步的重掩码（恢复哪些 token），如此迭代直至全部恢复。它们可叠加用在任意重掩码策略（Low-conf/Entropy/Margin）上，无需改模型、无需重训。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：图像 + 问题 prompt"] --> B["dMLLM 扩散去噪步"]
    B --> C["位置-步骤惩罚 PSP<br/>早期压低末端 token 置信度"]
    B --> D["视觉推理引导 VRG<br/>CFG 放大视觉条件 logits"]
    C --> E["重掩码：恢复高置信 token"]
    D --> E
    E -->|未完成，进入下一时间步| B
    E -->|全部恢复| F["输出答案"]
```

### 关键设计

**1. 位置-步骤惩罚 (PSP)：早期时间步压住序列末端的答案 token**

答案通常落在序列末端，而 dMLLM 偏偏在扩散早期就把末端 token 定了下来。PSP 据此在早期对越靠末端的 token 施加越强的惩罚：

$$\tilde{C}_j^i = C_j^i \cdot [1 - \gamma(1-\tau_i)\text{rel}(j)]$$

其中 $\tau_i = i/K$ 是扩散进度、$\text{rel}(j)\in[0,1]$ 是 token 的相对位置、$\gamma$ 是惩罚强度。早期（$\tau_i$ 小）且越靠后（$\text{rel}(j)$ 大）的 token 被压得越狠，于是模型被迫先把中间推理补完、再去落答案。消融显示 PSP 确实把答案生成推迟到了更晚的时间步。

**2. 视觉推理引导 (VRG)：借 CFG 放大视觉条件信号**

dMLLM 初期对视觉 prompt 依赖太弱，VRG 把图像扩散里的 Classifier-Free Guidance 搬过来，在 logits 层面把"有视觉条件"相对"无条件"的差异放大：

$$\text{logits}_{vrg} = \text{logits}_u + (s_{vrg}+1) \cdot (\text{logits}_c - \text{logits}_u)$$

$\text{logits}_c$ 是条件于视觉 prompt 的输出、$\text{logits}_u$ 是无条件输出，$s_{vrg}$ 控制放大幅度。这相当于强行把模型的注意力拽回视觉信息上，单用时效果略优于 PSP，和 PSP 合用最佳。

### 损失函数 / 训练策略

完全免训练，仅在推理阶段生效。超参取 $\gamma=0.5$、$s_{vrg}=0.5$，统一用贪心解码以保证可复现。

## 实验关键数据

### 主实验

| 模型 | 方法 | M3CoT(64/32) | MMBench(64/32) | SQA-IMG(64/32) | V*Bench(64/32) |
|------|------|-------------|---------------|----------------|---------------|
| LaViDa | Low-conf | 45.8 | 72.8 | 71.0 | 42.9 |
| LaViDa | PSP+VRG | 48.4 | 74.9 | 72.8 | 45.5 |
| MMaDa | Low-conf | 33.7 | 56.1 | 56.4 | 35.6 |
| MMaDa | PSP+VRG | 36.3 | 59.9 | 56.9 | 38.2 |

### 消融实验

| 配置 | M3CoT | MMBench | SQA-IMG | V*Bench |
|------|-------|---------|---------|--------|
| Low-conf | 45.8 | 72.8 | 71.0 | 42.9 |
| +PSP | 47.6 | 74.3 | 72.0 | 44.5 |
| +VRG | 47.8 | 75.1 | 72.1 | 45.0 |
| +PSP+VRG | 48.4 | 74.9 | 72.8 | 45.5 |

### 关键发现
- PSP有效将答案生成推迟到较晚时间步
- VRG单独使用时效果略优于PSP，两者结合效果最佳
- L/T=64/32的PSP+VRG超越L/T=256/128的Low-conf，实现>3倍加速
- DDCoT和CCoT等AR-VLM的CoT方法在dMLLM上表现不佳，印证了dMLLM需要不同的推理增强策略
- 在不同重掩码策略(Low-conf/Entropy/Margin)上均有效果

## 亮点与洞察
- 首次对dMLLM的推理过程进行定量分析，两个discovering非常有启发性
- AR-VLM vs dMLLM的视觉依赖模式对比揭示了本质性差异
- PSP的设计直觉简洁有效：位置×步骤的双重惩罚完美匹配问题
- VRG将CFG从图像扩散迁移到语言扩散的视觉推理，是自然且有效的类比

## 局限与展望 / 可改进方向
- VRG需要额外一次无条件前向传播（虽可并行），增加了计算开销
- γ和$s_{vrg}$固定为0.5，可能非最优；自适应策略值得探索
- 分析主要基于M3CoT数据集，后续需在更多推理场景（如视觉数学、图表理解）上验证泛化性
- dMLLM本身推理能力仍弱于AR-VLM，本方法是缓解而非根治

## 相关工作与启发
- 与AR-CoT相比，扩散CoT的核心在于重掩码策略而非序列生成
- ICoT等在AR-VLM上有效的方法在dMLLM上失效，突显了范式差异
- 对于未来dMLLM推理研究有重要参考价值：需要专门为并行生成设计的推理增强方法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次分析dMLLM推理，两个发现和对应方法都新颖
- 实验充分度: ⭐⭐⭐⭐ 双模型验证，多基准多配置，完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 分析→问题→方法的逻辑链非常自然
- 价值: ⭐⭐⭐⭐ 对新兴的dMLLM研究方向有重要指导意义

## 补充说明
- LaViDa基于LLaDA + reasoning微调，MMaDa基于8B MixCoT
- PSP和VRG可组合使用于任意重掩码策略（Low-conf/Entropy/Margin），均有改善
- VRG需要额外一次无条件前向传播，但可与条件前向并行计算
- M3CoT覆盖科学/数学/常识多个推理领域，是评估CoT推理的综合基准
- MMaDa在PSP+VRG下MMBench从56.1提升至59.9，绝对提升3.8%
- 所有实验不使用温度缩放，采用贪心解码以保证可复现性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](llada-v_large_language_diffusion_models_with_visual_instruction_tuning.md)
- [\[CVPR 2026\] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](sparse-lavida_sparse_multimodal_discrete_diffusion_language_models.md)
- [\[CVPR 2026\] Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation](guiding_diffusion-based_reconstruction_with_contrastive_signals_for_balanced_vis.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/multimodal_vlm/conditional_diffusion_sampling.md)
- [\[CVPR 2026\] Diffusion Guided Chain-of-Vision for Large Autoregressive Vision Models](diffusion_guided_chain-of-vision_for_large_autoregressive_vision_models.md)

</div>

<!-- RELATED:END -->
