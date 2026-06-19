---
title: >-
  [论文解读] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection
description: >-
  [AAAI 2026][AI安全][深度伪造检测] 提出 DFF-Adapter（DeepFake Fine-Grained Adapter），针对 DINOv2 设计的轻量级深度伪造检测微调方案。通过在每个 Transformer 块中注入三分支适配器（真实性检测头、伪造类型分类头、共享头），结合 Forgery-Aware Multi-Head Router 实现子空间级 LoRA 专家动态路由，利用辅助的伪造类型分类任务增强主任务的伪影敏感性，仅 3.5M 可训练参数即在多个跨数据集评估中达到 SOTA。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "深度伪造检测"
  - "DINOv2"
  - "参数高效微调"
  - "细粒度分类"
  - "LoRA"
---

# Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection

**会议**: AAAI 2026  
**arXiv**: [2511.12107](https://arxiv.org/abs/2511.12107)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 深度伪造检测, DINOv2, 参数高效微调, 细粒度分类, LoRA

## 一句话总结

提出 DFF-Adapter（DeepFake Fine-Grained Adapter），针对 DINOv2 设计的轻量级深度伪造检测微调方案。通过在每个 Transformer 块中注入三分支适配器（真实性检测头、伪造类型分类头、共享头），结合 Forgery-Aware Multi-Head Router 实现子空间级 LoRA 专家动态路由，利用辅助的伪造类型分类任务增强主任务的伪影敏感性，仅 3.5M 可训练参数即在多个跨数据集评估中达到 SOTA。

## 研究背景与动机

### 深度伪造检测的挑战

深度伪造技术迅速发展，合成内容越来越逼真，传统取证方法难以应对。核心挑战是**泛化性**——在一个数据集训练的检测器往往无法有效检测来自其他数据集或未知伪造方法的样本。

### 现有检测方法的局限

**传统方法**：生理/物理伪影检测、噪声残差分析、特征一致性分析等对特定伪造方法依赖强，泛化有限

**基于预训练大模型的方法**：
   - 微调范围不足：仅在最后一个/几个 Transformer 块插入适配器
   - 缺乏任务特定设计：将深度伪造检测简单当作二分类任务，忽略了不同伪造方法产生的**不同伪影模式**

### 为什么选择 DINOv2

与 CLIP（图像-文本对齐、侧重语义）不同，DINOv2 采用纯视觉自监督学习范式，更好地保留了细粒度的**局部纹理和几何结构**，对微妙的伪造痕迹更敏感。

### 核心动机

**不同伪造方法产生不同的伪影模式**，这些方法特异性线索是有信息量但被低估的。如果模型不仅判断"真/假"，还能识别"哪种伪造方法"，那么它必然对各类伪影更敏感。因此，**伪造类型分类**可作为辅助任务增强**真实性判别**。

## 方法详解

### 整体框架

在冻结的 DINOv2 骨干每个 Transformer 块中注入 DFF-Adapter。训练阶段三分支结构：
1. **真实性检测分支**：二分类 BCE 损失
2. **伪造类型分类分支**：多分类交叉熵损失
3. **共享分支**：在两个任务间传递细粒度伪造线索

推理阶段：仅使用融合了共享分支的真实性检测分支。

### 关键设计

#### 1. **Forgery-Aware Multi-Head Router（DF-MHR）**

核心组件，实现子空间级 LoRA 专家动态路由。

**通道分割**：隐藏状态 $\mathbf{X} \in \mathbb{R}^{L \times d}$ 沿通道分为 $h$ 个头适配器。

**共享 LoRA 专家池**：$N$ 个低秩 LoRA 专家 $\{(\mathbf{A}_j, \mathbf{B}_j)\}$，所有头适配器共享。

**任务特定路由**：对任务 $t$ 的第 $k$ 个头适配器，通过 softmax + Top-3 路由：
$$f_t^k = \beta \sum_{j \in S_{t,k}} \tilde{g}_{t,k}^{(j)} \mathbf{B}_j(\mathbf{A}_j \mathbf{X}^{(k)})$$
$$S_{t,k} = \text{Top3}(N, g_{t,k}), \quad g_{t,k} = \sigma(\mathbf{Z}_{task}[t,k])$$

**共享头适配器**：使用全局路由，所有 $N$ 个专家软加权。

设计动机：不同通道子空间编码不同方面的伪造特征，分割后各子空间可独立路由到最适合的专家。Top-3 路由比全部加权更稀疏高效。

#### 2. **Shared-Enhanced Task Fusion**

通过残差融合跨所有 Transformer 块传递共享和任务特定更新：
$$f_{bin} = h_{cls} + \text{concat}(f_{share}, f_0^1, ..., f_0^n)$$
$$f_{ftc} = h_{cls} + \text{concat}(f_{share}, f_1^1, ..., f_1^n)$$

设计动机：共享分支是桥梁——当伪造类型分类任务迫使共享分支学习方法特异性伪影特征时，这些特征自动流入真实性分支。

#### 3. **双任务解耦训练**

每个 mini-batch 进行两次前向传播（task flag=0 真实性 + task flag=1 伪造类型），避免梯度干扰：
$$\mathcal{L} = \lambda_0 \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{ftc}$$

### 损失函数 / 训练策略

- 骨干冻结（DINOv2-Large-with-registers），仅训练 DFF-Adapter
- 可训练参数：**仅 3.5M**
- 配置：rank $r=16$, $\alpha=32$, 6 个 LoRA 专家, 4 个头适配器
- Adam 优化器，lr=$2 \times 10^{-4}$，50 epochs
- 损失权重：$\lambda_0 = 10$, $\lambda_1 = 2$

## 实验关键数据

### 主实验

**跨数据集评估（训练于 FF++ c23，AUC %）**：

| 方法 | 会议 | CDF-v2 | DFDC | CDF-v1 | DFDCP | 平均 |
|------|------|--------|------|--------|-------|------|
| SBI | CVPR'22 | 93.82 | 74.47 | 93.44 | 90.95 | 88.17 |
| LVLM-DFD | ICML'25 | 94.71 | 79.12 | 97.62 | 91.81 | 90.82 |
| VB-StA | CVPR'25 | 94.7 | 84.3 | - | 90.9 | - |
| UDD | AAAI'25 | 93.13 | 81.21 | - | 88.11 | - |
| **DFF-Adapter** | - | **95.26** | **89.96** | 96.14 | **91.57** | **93.23** |

平均 AUC 93.23%，比 LVLM-DFD 高 2.41 个点。DFDC 上提升 +10.84。

**跨伪造方法评估（DF40）**：

| 方法 | FaceDancer | InSwapper | FSGAN | HyperReenact | Wav2Lip | DiT-XL/2 |
|------|-----------|-----------|-------|-------------|---------|----------|
| LVLM-DFD | 82.97 | 87.64 | 93.75 | 81.56 | 78.60 | 86.61 |
| **DFF-Adapter** | **93.15** | **93.98** | **98.84** | **90.28** | **91.65** | **98.44** |

所有伪造方法均取得最优。

### 消融实验

**核心组件的影响（AUC %）**：

| 配置 | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|------|--------|------|--------|-------|
| 仅 DINOv2 线性探测 | 61.93 | 52.83 | 68.46 | 55.54 |
| + FAMHR | 89.56 | 86.24 | 85.53 | 87.11 |
| **+ FAMHR + SETF** | **95.26** | **89.96** | **96.14** | **91.57** |

**与其他微调策略对比**：

| 微调方式 | CDF-v2 | DFDC | DFDCP |
|----------|--------|------|-------|
| Linear Probing | 66.63 | 72.54 | 65.90 |
| LoRA | 85.14 | 79.95 | 81.49 |
| MoE-FFD | 80.40 | 76.52 | 87.60 |
| **DFF-Adapter** | **95.26** | **89.96** | **91.57** |

**身份受限训练（极少身份数下的泛化）**：

| 训练身份数 | CDF-v2 | DFDC | CDF-v1 | DFDCP |
|-----------|--------|------|--------|-------|
| 10 (~1%) | 81.97 | 82.63 | 81.62 | 82.22 |
| 50 (~5%) | 87.28 | 82.37 | 91.26 | 86.16 |

仅 10 个身份即可维持竞争力，展现极强数据效率。

### 关键发现

1. **在所有 Transformer 块注入适配器**至关重要，允许浅层参与任务特定学习
2. **伪造类型辅助监督**提供超越二分类的细粒度伪影信息
3. **DINOv2 比 CLIP 更适合伪造检测**
4. **t-SNE 可视化**证实 DFF-Adapter 形成了按伪造类型区分的子簇

## 亮点与洞察

1. **辅助任务设计精妙**：伪造类型分类不仅是辅助任务，更是隐式数据增强
2. **共享分支的桥梁作用**：推理时无需伪造类型分支的额外开销
3. **参数效率极高**：3.5M 参数，单卡 4090 训练
4. **DFDC 上 +10.84 的提升**：在最困难基准上证明鲁棒性

## 局限与展望

1. 依赖 FF++ 训练，未探索更大训练集
2. 训练需伪造类型标签，某些实际场景可能不可用
3. 双次前向传播增加训练计算成本
4. 仅图像级检测，未做像素级伪造区域定位

## 相关工作与启发

- **DINOv2**：视觉基础模型骨干，自监督保留丰富局部特征
- **LoRA 系列**：参数高效微调基础
- **MoE-FFD**：最直接对比，但缺乏任务特定设计
- **LVLM-DFD**：利用大语言模型的强基线
- **SBI**：自混合训练实现泛化

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双任务监督+共享分支设计有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 跨数据集、跨伪造方法、身份受限等多维度评估
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，消融详尽
- **实用价值**: ⭐⭐⭐⭐⭐ — 3.5M 参数、单卡训练、SOTA 性能，直接可部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization](../../CVPR2026/ai_safety/diffusionff_a_diffusion-based_framework_for_joint_face_forgery_detection_and_fin.md)
- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](../../CVPR2025/ai_safety/towards_general_visual-linguistic_face_forgery_detection.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](../../ICML2026/ai_safety/towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->
