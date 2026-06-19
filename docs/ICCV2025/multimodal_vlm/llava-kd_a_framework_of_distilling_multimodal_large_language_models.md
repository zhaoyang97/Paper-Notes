---
title: >-
  [论文解读] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models
description: >-
  [ICCV2025][多模态VLM][知识蒸馏] 提出 LLaVA-KD 框架，通过多模态蒸馏（MDist）和关系蒸馏（RDist）两种策略，结合三阶段训练方案（DPT-SFT-DFT），将大规模 MLLM 的知识高效迁移至小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。 多模态大语言模型（MLLM）在统一理解…
tags:
  - "ICCV2025"
  - "多模态VLM"
  - "知识蒸馏"
  - "多模态大模型"
  - "小模型训练"
  - "视觉-语言对齐"
---

# LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

**会议**: ICCV2025  
**arXiv**: [2410.16236](https://arxiv.org/abs/2410.16236)  
**代码**: [GitHub](https://github.com/Fantasyele/LLaVA-KD)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 多模态大模型, 小模型训练, 视觉-语言对齐  

## 一句话总结

提出 LLaVA-KD 框架，通过多模态蒸馏（MDist）和关系蒸馏（RDist）两种策略，结合三阶段训练方案（DPT-SFT-DFT），将大规模 MLLM 的知识高效迁移至小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。

## 研究背景与动机

多模态大语言模型（MLLM）在统一理解视觉和语言信息方面取得了巨大成功，但模型规模的持续增长限制了其在资源受限场景下的部署。现有的小规模 MLLM（s-MLLM）通常采用轻量级 LLM 骨干网络来降低计算成本，但直接沿用大模型的两阶段训练范式（PT → SFT）会导致显著的性能下降。例如，4B 的 TinyLLaVA 可达 65.0%，但降至 0.5B 后性能骤降至 54.7%。

已有工作尝试通过以下方式弥补：
- **模型结构优化**：MoE-LLaVA 引入混合专家结构
- **训练数据优化**：Bunny 通过聚类+剪枝提高数据质量

但这些方法要么引入额外参数，要么增大数据成本。作者认为，**训练范式优化**是一条被忽视但极具潜力的路径。现有 LLM 蒸馏方法仅关注文本模态的知识迁移，忽略了视觉模态在多模态理解中的关键作用，且直接在 SFT 阶段引入蒸馏收益有限。

## 方法详解

### 整体框架

LLaVA-KD 包含一个大规模教师模型（l-MLLM）和一个小规模学生模型（s-MLLM），两者均采用 LLaVA-1.5 架构（Visual Encoder + Projector + LLM）。教师和学生共享相同的视觉编码器（SigLIP-B/14@384px），通过两层 MLP 投影器将视觉特征 $Z_v \in \mathbb{R}^{N_p \times C}$ 映射至文本嵌入空间 $H_v \in \mathbb{R}^{N_p \times D}$。

### 关键设计一：多模态蒸馏（MDist）

MDist 同时在**响应 token** 和**视觉 token** 两个维度进行 KL 散度蒸馏：

**响应蒸馏**：对齐教师和学生在响应 token 上的输出分布：

$$\mathcal{L}_{res} = \sum_{m=1}^{M} \text{KLD}(\phi_l(y_m | \mathbf{y}_{<m}), \phi_s(y_m | \mathbf{y}_{<m}))$$

**视觉蒸馏**：对齐教师和学生在视觉 token 上的输出分布：

$$\mathcal{L}_{vis} = \sum_{k=1}^{K} \sum_{j=1}^{V} \phi_l(Y_j | \mathbf{y}_{<k}) \log \frac{\phi_l(Y_j | \mathbf{y}_{<k})}{\phi_s(Y_j | \mathbf{y}_{<k})}$$

其中 $K$ 为视觉 token 长度，$V$ 为词表大小。与仅蒸馏响应 token 的传统 LLM 蒸馏不同，MDist 显式地将视觉模态纳入蒸馏范围，确保多模态表示的全面迁移。

### 关键设计二：关系蒸馏（RDist）

RDist 通过构建视觉 token 的自相关矩阵来迁移教师模型捕获视觉 token 间关系的能力。具体地，分别计算教师和学生的自相关矩阵：

$$R_v^s = \mathbf{y}_v^s \otimes \mathbf{y}_v^s \in \mathbb{R}^{N_p \times N_p}, \quad R_v^t = \mathbf{y}_v^t \otimes \mathbf{y}_v^t \in \mathbb{R}^{N_p \times N_p}$$

然后通过最大化余弦相似度来对齐两者：

$$\mathcal{L}_{rel} = 1 - \text{Cos}(R_v^s, R_v^t) = 1 - \frac{R_v^s \cdot R_v^t}{\|R_v^s\| \|R_v^t\|}$$

这种设计编码了视觉 token 之间的空间和语义依赖关系（如物体位置、交互关系），对复杂视觉场景理解至关重要。

### 三阶段训练方案

1. **Distilled Pre-Training（DPT）**：冻结视觉编码器和 LLM，仅训练投影器。在标准自回归损失基础上加入 MDist 和 RDist：$\mathcal{L}_{DPT} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{res} + \beta \mathcal{L}_{vis} + \gamma \mathcal{L}_{rel}$，增强视觉-文本对齐质量。

2. **Supervised Fine-Tuning（SFT）**：标准 SFT，联合训练投影器和 LLM，建立基础多模态理解能力。

3. **Distilled Fine-Tuning（DFT）**：在 SFT 后再次引入蒸馏，精炼学生模型的知识：$\mathcal{L}_{DFT} = \mathcal{L}_{reg} + \alpha' \mathcal{L}_{res} + \beta' \mathcal{L}_{vis} + \gamma' \mathcal{L}_{rel}$

所有损失权重 $\{\alpha, \beta, \gamma\}$ 和 $\{\alpha', \beta', \gamma'\}$ 均设为 1.0, 1.0, 0.5。

## 实验结果

### 主实验：与 SoTA 方法对比

| 方法 | LLM | VQAv2 | GQA | SciQA | MME | MMB | POPE | Avg₁₀ |
|------|-----|-------|-----|-------|-----|-----|------|-------|
| TinyLLaVA (Qwen1.5-0.5B) | 0.5B | 73.9 | 57.4 | 60.9 | 59.8 | 55.0 | 83.7 | 54.7 |
| LLaVA-MOD (Qwen1.5-0.5B) | 0.5B | - | 56.2 | 62.8 | 65.3 | 58.8 | - | 54.1 |
| **LLaVA-KD (Qwen1.5-0.5B)** | **0.5B** | **77.0** | **59.6** | **60.6** | **64.5** | **60.1** | **85.9** | **57.9** |
| TinyLLaVA (Qwen1.5-1.8B) | 1.8B | 73.1 | 55.5 | 65.3 | 61.2 | 57.1 | 83.4 | 56.8 |
| LLaVA-MOD (Qwen1.5-1.8B) | 1.8B | - | 58.7 | 68.0 | 66.7 | 66.3 | 87.0 | 59.9 |
| **LLaVA-KD (Qwen1.5-1.8B)** | **1.8B** | **79.0** | **62.3** | **64.7** | **69.1** | **64.0** | **86.3** | **62.1** |

LLaVA-KD 在 0.5B 和 1.8B 规模上均超越基线，在 Avg₁₀ 上分别提升 3.2% 和 5.3%，且仅需 1.2M 训练样本（LLaVA-MOD 需 5M）。

### 消融实验：三阶段训练方案的效果

| 训练方案 | Avg₁₀ |
|---------|-------|
| PT-SFT（基线） | 54.7 |
| DPT-SFT | 55.6 (+0.9) |
| PT-DFT | 55.8 |
| DPT-DFT | 55.9 |
| PT-SFT-DFT | 56.6 |
| **DPT-SFT-DFT** | **57.9 (+3.2)** |
| DPT-DFT-DFT | 58.0 |

- DPT 带来 0.9% 提升，说明蒸馏式预训练增强了跨模态对齐
- DFT 贡献最大（+2.3%），说明其有效迁移了教师知识
- 跳过 SFT 阶段（DPT-DFT）性能下降，证明 SFT 对知识习得不可或缺
- DPT-DFT-DFT 性能略优但计算开销增大（120 GPU hours），DPT-SFT-DFT 是最佳性价比方案

### 蒸馏目标消融

| 蒸馏目标 | Response | Visual | Avg₁₀ |
|---------|----------|--------|-------|
| DPT: 仅 Response | ✓ | ✗ | 54.9 |
| DPT: Response + Visual | ✓ | ✓ | 55.1 |
| DFT: 仅 Response | ✓ | ✗ | 57.2 |
| DFT: Response + Visual | ✓ | ✓ | **57.7** |

在 DPT 和 DFT 阶段，对视觉 token 的蒸馏均带来额外提升，验证了 MDist 中视觉蒸馏的重要性。

## 亮点与洞察

1. **多模态蒸馏思路新颖**：首次将蒸馏从响应 token 扩展至视觉 token，弥补了现有 LLM 蒸馏方法忽略视觉模态的缺陷
2. **关系蒸馏设计巧妙**：通过视觉 token 自相关矩阵捕获空间和语义关系，而非简单的特征对齐
3. **三阶段训练方案有据可循**：DPT 增强对齐、SFT 建立基础、DFT 精炼知识，每个阶段都有明确的功能定位
4. **架构无关性强**：无需修改模型架构，可直接应用于各种 LlaVA 风格的 MLLM

## 局限性

- 教师和学生必须共享相同的视觉编码器，限制了蒸馏的灵活性
- 蒸馏增加了训练计算和显存开销（需要同时运行教师和学生模型）
- 仅在 LLaVA-1.5 架构上验证，对更先进的架构（如动态分辨率）的适用性未知
- 损失权重（α, β, γ）固定为经验值，缺乏自适应调整机制

## 相关工作

- **小规模 MLLM**：TinyLLaVA, Bunny, MoE-LLaVA, MobileVLM, MiniCPM-V 等通过轻量骨干或结构优化降低成本
- **LLM 蒸馏**：MiniLLM（反向 KLD）、DistiLLM（偏斜 KLD）、CoT 蒸馏等聚焦文本模态
- **MLLM 蒸馏**：LLaVA-MoD（输出 KLD + 偏好蒸馏 + MoE）、LLaVADI 发现多数 LLM 蒸馏策略对 MLLM 无额外收益

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LLaVA-Critic: Learning to Evaluate Multimodal Models](../../CVPR2025/multimodal_vlm/llava-critic_learning_to_evaluate_multimodal_models.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](../../ACL2025/multimodal_vlm/hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](../../ACL2025/multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)
- [\[ICLR 2026\] LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](../../ICLR2026/multimodal_vlm/llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
