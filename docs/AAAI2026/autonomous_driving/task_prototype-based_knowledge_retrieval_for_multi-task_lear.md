# Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data

**会议**: AAAI 2026  
**arXiv**: [2601.07474](https://arxiv.org/abs/2601.07474)  
**代码**: 暂无  
**领域**: 多任务学习 / 密集预测 / 部分标注  
**关键词**: Multi-Task Learning, Task Prototype, 部分标注, Knowledge Retrieval Transformer, 任务关联

## 一句话总结

提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

## 研究背景与动机

自动驾驶和机器人等应用需要同时处理语义分割、深度估计、法线估计等多个密集预测任务。但为所有任务获取完整标注成本极高，因此部分标注多任务学习（MTPSL）成为实际需求。

现有 MTPSL 方法的问题：
- **MTPSL (Li et al., CVPR'22)**：通过联合任务空间映射实现跨任务正则化，但依赖未标注任务的预测，噪声大
- **DiffusionMTL (Ye & Xu, CVPR'24)**：用扩散模型结合多任务条件整合跨任务信息，但预测型和特征型输入导致性能不一致
- **共同局限**：都依赖未标注任务的预测来建立任务关联——而未标注预测本身含噪且不完整，导致负迁移

核心问题：如何在仅有部分标注的条件下，不依赖未标注任务的预测也能建立可靠的任务关联，并自适应优化各任务的特征表示？

## 方法详解

### 整体框架

两大模块端到端联合训练：
1. **多任务学习模块**：Backbone (ResNet-18) → 编码特征 f^e → Vector Quantization 增强共享表示 → 各任务解码器 → 任务特征 f^t
2. **原型知识检索模块**：f^t → Task Prototype 𝒱 生成 task-affinity score → Knowledge Retrieval Transformer 基于 affinity 自适应精炼 → 任务精炼特征 f^{tr} → 各任务预测头

### 关键设计

1. **Vector Quantization 增强表示**：
   - 维护可学习码本 𝒵 = {z_k}_{k=1}^K（K=4096 个 slot，维度 1024）
   - 编码特征 f^e 通过最近邻量化映射到码本得到 f^q，与 f^e 逐元素求和得到 f^i
   - 通过 Task-Agnostic Enhancement (TAE) Loss 训练码本：让 f^i 经卷积解码器重建输入图像，用 Smooth L1 损失
   - 目的：在部分标注下扩展共享特征空间，使各任务线索都能被保留

2. **Task Prototype 𝒱**：
   - 包含 T 个可学习 slot（T=任务数，PASCAL-Context T=5，NYUD-v2 T=3），每个 v_τ ∈ R^{1×d}（d=1024）
   - 计算 task-similarity：S(f̂^t, 𝒱) = cosine_similarity(f̂^t, v_τ) 对各任务 τ
   - **Task Knowledge Embedding (TKE) Loss**：对 softmax 后的 affinity score 做交叉熵，目标是 one-hot 向量 Y_t
   - **Task Consistency (TC) Loss**：批内聚合同一任务的特征 x̃^t，用 triplet loss 确保同任务特征高相似、异任务特征低相似
   - Association Knowledge Generating (AKG) Loss = ℒ_tke + ℒ_tc

3. **Knowledge Retrieval Transformer**：
   - 输入：任务特征 f̂^t 和 task-affinity feature f^{ta} = 𝒜(f̂^t, 𝒱) · 𝒱
   - 多个 knowledge-retrieval block：Self-Attention → Cross-Attention (query=f^t, key/value=f^{ta}) → FFN
   - Cross-Attention 中 f^{ta} 编码了"当前任务需要从哪些其他任务获取多少增强"的信息
   - 输出：任务精炼特征 f^{tr}，送入各任务预测头

### 损失函数

总损失：ℒ_Total = ℒ_MTL + λ₁·Σℒ_tae + λ₂·ℒ_akg

- ℒ_MTL：有标注样本上的监督损失（语义分割/parsing/saliency/boundary→交叉熵，depth/normal→L1）
- ℒ_tae：码本增强的图像重建损失（Smooth L1）
- ℒ_akg = ℒ_tke + ℒ_tc：确保原型嵌入正确的任务特性

训练细节：Adam，lr=2e-5，PASCAL-Context 100 epochs batch=6，NYUD-v2 200 epochs batch=4，单 RTX A6000

## 实验关键数据

### 主实验：PASCAL-Context（5 任务）

| 方法 | 设置 | Semseg mIoU↑ | Parsing mIoU↑ | Saliency maxF↑ | Normal mErr↓ | Boundary odsF↑ |
|------|------|:-:|:-:|:-:|:-:|:-:|
| Single-Task | one-label | 50.34 | 59.05 | 77.43 | 16.59 | 64.40 |
| MTL Baseline | one-label | 44.73 | 57.03 | 75.69 | 16.47 | 64.30 |
| MTPSL* (CVPR'22) | one-label | 55.08 | 56.72 | 77.06 | 16.93 | 63.70 |
| DiffusionMTL-Pred (CVPR'24) | one-label | 59.43 | 56.79 | 77.57 | 16.20 | 64.00 |
| DiffusionMTL-Feat (CVPR'24) | one-label | 57.78 | 58.98 | 77.82 | 16.11 | 64.50 |
| **本文方法** | **one-label** | **59.78** | **59.08** | **78.62** | **15.63** | **65.10** |
| MTPSL* (CVPR'22) | random | 62.44 | 55.81 | 78.56 | 15.45 | 66.80 |
| DiffusionMTL-Feat (CVPR'24) | random | 62.55 | 56.84 | 80.44 | 14.85 | 67.10 |
| **本文方法** | **random** | **64.30** | **56.87** | **80.51** | **14.48** | **67.30** |

### NYUD-v2（3 任务）

| 方法 | 设置 | Semseg mIoU↑ | Depth absErr↓ | Normal mErr↓ |
|------|------|:-:|:-:|:-:|
| Single-Task | one-label | 45.28 | 0.4802 | 25.93 |
| MTPSL* (CVPR'22) | one-label | 43.97 | 0.5140 | 26.30 |
| DiffusionMTL-Feat (CVPR'24) | one-label | 44.47 | 0.5059 | 25.84 |
| **本文方法** | **one-label** | **45.95** | **0.4865** | **25.64** |
| **本文方法** | **random** | **47.53** | **0.4621** | **24.67** |

### 消融实验：各损失函数贡献（PASCAL-Context one-label）

| ℒ_tae | ℒ_tke | ℒ_tc | Semseg | Parsing | Saliency | Normal↓ | Boundary |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | 44.73 | 57.03 | 75.69 | 16.47 | 64.38 |
| ✓ | ✗ | ✗ | 44.83 | 57.13 | 76.13 | 16.22 | 64.50 |
| ✓ | ✓ | ✗ | 58.21 | 58.87 | 78.50 | 15.67 | 65.00 |
| ✓ | ✓ | ✓ | **59.78** | **59.08** | **78.62** | **15.63** | **65.10** |

### 原型维度消融（NYUD-v2 one-label）

| 维度 | 参数量 | Semseg↑ | Depth↓ | Normal↓ |
|------|:-:|:-:|:-:|:-:|
| 无原型 | 146.5M | 42.77 | 0.5134 | 26.99 |
| 256 | 156.6M | 44.91 | 0.4931 | 25.67 |
| 512 | 157.5M | 45.65 | 0.4878 | 25.73 |
| **1024** | **159.4M** | **45.95** | **0.4865** | **25.64** |
| 2048 | 163.0M | 45.33 | 0.4867 | 25.77 |

### 关键发现

- ℒ_tke 是最大贡献者（从 44.83→58.21 提升 Semseg ~13 个点），说明任务原型嵌入任务知识的有效性
- ℒ_tc 进一步提升一致性（58.21→59.78），确保不同样本下任务特性一致
- 原型维度 1024 最优——太小无法嵌入足够特性，太大反而难利用信息
- 与 prompt-based 方法（TaskPrompter, TSP-Transformer）对比：本文用 explicit learning 显式嵌入任务特性，在部分标注下优于依赖监督的 latent learning（50.08 vs 48.68 Semseg on NYUD-v2）

## 亮点与洞察

- **核心创新**：用显式的 Task Prototype 替代对未标注任务预测的依赖——从"伪标签驱动"转向"任务特性驱动"
- Task-affinity score 提供了可解释的跨任务关联可视化：不同目标任务激活不同的原型 slot
- Knowledge Retrieval Transformer 的 cross-attention 实现了"按需检索"——每个任务只获取相关的跨任务知识
- Vector Quantization 扩展共享特征空间的思路来自 VQ-VAE，巧妙适配到部分标注场景

## 局限性

- 仅在 ResNet-18 骨干上验证，未测试更大模型（ViT-L 仅在 Discussion 中与 prompt 方法对比）
- 任务原型为固定 slot 数（= 任务数），未探索过参数化对泛化的影响
- 需要所有任务的标注都至少部分可用（one-label 设置仍要求每张图片有一个任务标签）
- 消融中 TAE Loss 贡献较小（44.73→44.83），码本的必要性存疑

## 相关工作与启发

- Task Prototype 思想可推广到多模态学习（不同模态作为不同"任务"）
- AKG Loss 的设计（TKE + TC）可作为通用的原型学习正则化方案
- 对比 DiffusionMTL：扩散模型在 MTL 中的效果依赖输入类型（prediction vs feature），不够稳健

## 评分

- 新颖性：⭐⭐⭐⭐（Task Prototype + Knowledge Retrieval 的组合新颖）
- 技术深度：⭐⭐⭐⭐（损失设计层次清晰，VQ+Prototype+Transformer 三组件协同）
- 实验充分度：⭐⭐⭐⭐⭐（两数据集+多消融+维度分析+prompt 方法对比+可视化）
- 实用价值：⭐⭐⭐⭐（部分标注场景普遍存在，方法通用性好）
- 任务数量较多时关联矩阵计算成本增加
- 动态任务权重策略可进一步优化

## 与相关工作的对比
- vs 伪标签方法：避免噪声标签导致的负迁移
- vs 标准 MTL：支持部分标注场景
- vs 任务分组方法：通过原型自适应学习任务关联

## 启发与关联
任务原型的概念可推广至其他需要量化任务关系的场景。AKG Loss 确保无监督环境下原型质量的设计有参考价值。

## 评分 ⭐⭐⭐⭐ (4/5)
问题重要且实际，框架设计合理。对自动驾驶中的多任务感知有直接应用场景。信息有限，评分保守。
