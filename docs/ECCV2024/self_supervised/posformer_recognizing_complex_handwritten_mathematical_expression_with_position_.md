---
title: >-
  [论文解读] PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer
description: >-
  [ECCV 2024][自监督学习][手写数学公式识别] 提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "手写数学公式识别"
  - "位置森林"
  - "Transformer"
  - "注意力校正"
  - "结构关系建模"
---

# PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer

**会议**: ECCV 2024  
**arXiv**: [2407.07764](https://arxiv.org/abs/2407.07764)  
**代码**: [有](https://github.com/SJTU-DeepVisionLab/PosFormer)  
**领域**: 手写数学公式识别 / 序列识别  
**关键词**: 手写数学公式识别, 位置森林, Transformer, 注意力校正, 结构关系建模

## 一句话总结

提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。

## 研究背景与动机

### 手写数学公式识别的挑战

手写数学公式识别（HMER）旨在将表达式图像转换为 LaTeX 序列，广泛应用于在线教育、手稿数字化、自动评分等场景。其核心难点在于：

**符号间关系复杂**：数学公式包含上下标、分数、根号等嵌套结构，符号间的空间位置关系（上/中/下）和层级关系（嵌套深度）比普通文本复杂得多。模型需要正确生成 `^`, `_`, `{`, `}` 等结构符号来描述这些关系。

**手写风格多样性**：不同人的书写在尺度、倾斜度、连笔方式上差异巨大，增加了识别难度。

### 现有方法的局限

**树结构方法**（如 BLSTM、SAN、TSDNet）：将表达式建模为语法树，输出 (父节点, 子节点, 关系) 三元组。但树结构在不同表达式间多样性不足，泛化能力差，精度较低。

**序列方法**（如 WAP、BTTR、CoMER、LAST）：将 HMER 建模为端到端的 image-to-sequence 任务，使用注意力编码器-解码器架构自回归预测 LaTeX 序列。这类方法更灵活且精度更高，但只能**隐式**学习 LaTeX 的结构规则——当面对复杂嵌套表达式时，模型难以正确理解深层嵌套中符号的相对位置关系。

**注意力漂移问题**：CoMER 通过减去所有已解码步骤的注意力来校正当前注意力。但在解码结构符号（如 `^`, `{`）时，模型会关注尚未解析的区域以理解结构关系。这些注意力被累积为校正项后，会导致后续实体符号解码时注意力偏移错误。例如解码 `4^{x-\frac{1}{4}}` 时，`^` 和 `{` 的注意力会干扰后续 `x` 的正确关注区域。

### PosFormer 的设计动机

PosFormer 的核心洞察是：**位置信息可以从 LaTeX 序列本身解析出来，无需额外标注**。通过将数学表达式编码为"位置森林"结构（而非单棵树），可以同时处理独立和嵌套的子结构。将位置识别作为辅助任务联合优化，可以显式引导模型学习位置感知的特征表示，并且在推理时完全移除该分支，不增加任何计算开销。

## 方法详解

### 整体框架

PosFormer 采用编码器-解码器架构：
- **编码器**：DenseNet 骨干网络提取 2D 视觉特征
- **解码器**：三层 Transformer 解码器，包含多头注意力(MHA)、隐式注意力校正(IAC)和前馈网络(FFN)
- **输出头**：三个并行线性头分别预测 LaTeX 符号、嵌套层级、相对位置

训练时联合优化表达式识别和位置识别两个任务；推理时仅保留表达式识别分支。

### 关键设计

#### 1. 位置森林编码 (Position Forest Coding)

**功能**：将 LaTeX 序列编码为位置森林结构，为每个符号分配位置标识符（如 "MLLR"），表示其在二维空间中的相对位置。

**核心思路**：根据 LaTeX 语法规则，表达式可分为四类子结构（如图所示）：
- **上下标结构**：`^{...}` / `_{...}`
- **分数结构**：`\frac{...}{...}`
- **根号结构**：`\sqrt{...}`
- **特殊运算符**：如 Product、Limit 等

编码遵循三条规则：
1. 子结构按从左到右顺序编码
2. 每个子结构编码为一棵树：主体为根节点(M)，上部为左节点(L)，下部为右节点(R)
3. 根据子结构关系（独立或嵌套），将编码树串联或嵌套形成位置森林

例如，表达式 $A y_1^3 + \frac{y_2^{\beta_1}B}{C}$ 被分解为 8 个子结构，其中分数内部的子结构嵌套在分数树中。最终每个符号获得一个由 M/L/R 组成的位置标识符，如 `\beta` 的标识符为 "MLLR"（表示在主体的左-左-右位置，即第三层嵌套的下标位置）。

**设计动机**：与树结构方法不同，位置森林能自然处理独立子结构（多棵树并列）和嵌套子结构（树中嵌套树），更灵活地描述复杂公式的结构。关键优势在于不需要额外标注——位置信息完全从 LaTeX 序列的语法规则中自动推导。

#### 2. 位置森林解码

**功能**：将位置识别分解为两个子任务——嵌套层级预测和相对位置预测，作为辅助任务训练。

**核心思路**：给定位置标识符 $\mathbf{I}_k = \{q_k^{(i)} | q_k^{(i)} \in \{M, L, R\}\}_{i=1}^{\eta_k}$：
- **嵌套层级**为 $\eta_k - 1$（标识符长度减一）
- **相对位置**为 $q_k^{(\eta_k)}$（标识符最后一个字符）

例如，"MLLR" → 嵌套层级 = 3，相对位置 = R（下方）。

将标识符序列补齐为统一长度后，通过非线性层 $\xi$ 转换为标识符嵌入：

$$\mathbf{Q}_{\text{emb}} = [\xi(\mathbf{Q}_1); \xi(\mathbf{Q}_2); \cdots; \xi(\mathbf{Q}_T)] + \mathbf{Q}_{\text{pos}}$$

嵌入向量与视觉特征一起送入 Transformer 解码器，输出特征 $\mathbf{F}_t$ 分别用于预测嵌套层级和相对位置：

$$p(y_n^{(t)}) = \text{softmax}(\mathbf{W}_n\mathbf{F}_t + b_n)$$
$$p(y_r^{(t)}) = \text{softmax}(\mathbf{W}_r\mathbf{F}_t + b_r)$$

**设计动机**：将位置信息作为辅助监督信号，迫使解码器的中间特征编码位置感知信息。由于嵌套层级和相对位置的预测与符号识别共享特征表示，这种多任务学习隐式地改善了主任务的特征质量。在推理时移除该分支，不增加任何延迟。

#### 3. 隐式注意力校正 (Implicit Attention Correction, IAC)

**功能**：修正注意力权重的累积校正项，避免结构符号的注意力干扰后续实体符号的解码。

**核心思路**：关键观察——只有**实体符号**（如 `a`, `b`, `1`）在图像中有对应区域，**结构符号**（如 `^`, `_`, `{`, `}`）没有。因此，注意力校正项应只累积实体符号的注意力，结构符号的注意力用零替代。

引入指示函数 $I_\Omega$：

$$I_\Omega(y) = \begin{cases} \mathbf{1}, & \text{if } y \notin \Omega \\ \mathbf{0}, & \text{if } y \in \Omega \end{cases}$$

其中 $\Omega$ 为结构符号集合。校正项计算为：

$$\mathbf{A}_k^{(t)} = \sum_{i=1}^{t-1} (\tilde{\mathbf{E}}_k^{(i)} \odot I_\Omega(y_c^{(i)}))$$

最终校正注意力：

$$\hat{\mathbf{E}}_k = \mathbf{E}_k - \phi(\mathbf{A}_k)$$

**设计动机**：这是对 CoMER 注意力校正的精准修复。CoMER 的问题在于它不加区分地累积所有符号的注意力，但结构符号本身没有图像实体，其注意力往往指向未解析区域或全局图像。将这些"虚假"注意力排除在校正项之外，可以让后续解码步骤获得更准确的对齐信息。

### 损失函数 / 训练策略

**多任务损失**：

$$L_{\text{all}} = \lambda_1 \cdot L_{\text{rec}} + \lambda_2 \cdot L_{\text{pos}}$$

$$L_{\text{rec}} = -\frac{1}{T}\sum_{t=1}^{T} y_c^{(t)} \log p(y_c^{(t)})$$

$$L_{\text{pos}} = -\frac{1}{T}\sum_{t=1}^{T} (y_n^{(t)} \log p(y_n^{(t)}) + y_r^{(t)} \log p(y_r^{(t)}))$$

$\lambda_1 = \lambda_2 = 1$，两个任务等权重。

**训练设置**：
- 骨干网络：DenseNet
- 解码器：三层 Transformer
- 位置识别头：两个线性层（嵌套层级最大 3 级，相对位置词汇大小 6）
- 遵循 CoMER/LAST 的训练参数（batch size、学习率、优化器）
- NVIDIA A800 单卡训练

## 实验关键数据

### 主实验

**CROHME 单行数据集 (使用 scale augmentation)**

| 数据集 | 指标 | PosFormer | 之前SOTA (CoMER) | 提升 |
|--------|------|-----------|----------------|------|
| CROHME 2014 | ExpRate | **62.68%** | 59.33% | +2.03% (vs BPD-Coverage +2.03) |
| CROHME 2016 | ExpRate | **61.03%** | 59.81% | +1.22% |
| CROHME 2019 | ExpRate | **64.97%** | 62.97% | +2.00% |

**不使用 scale augmentation 的提升更大**

| 数据集 | PosFormer | 之前SOTA (CAN) | 提升 |
|--------|-----------|---------------|------|
| CROHME 2014 | **60.45%** | 57.26% | +3.19% |
| CROHME 2016 | **60.94%** | 56.15% | +4.79% |
| CROHME 2019 | **62.22%** | 56.34% | +5.88% |

**多行 M2E 数据集**

| 方法 | ExpRate ↑ | ≤1 ↑ | CER ↓ |
|------|-----------|------|-------|
| CoMER | 56.20% | 73.39% | 0.0499 |
| LAST | 56.50% | 72.80% | 0.0530 |
| **PosFormer** | **58.33%** | **75.58%** | **0.0366** |

提升 1.83%，CER 从 0.0499 降至 0.0366（显著降低）。

**大规模 HME100k 数据集**

| 方法 | ExpRate ↑ | ≤1 ↑ | ≤2 ↑ |
|------|-----------|------|------|
| CAN | 67.31% | 82.93% | 89.17% |
| **PosFormer** | **69.51%** | **84.91%** | **90.51%** |

### 消融实验

**各组件贡献 (CROHME 系列, ExpRate %)**

| 模型 | 2014 | 2016 | 2019 |
|------|------|------|------|
| baseline (CoMER) | 59.33 | 59.81 | 62.97 |
| + PF | 62.13 (+2.80) | 61.03 (+1.22) | 63.80 (+0.83) |
| + PF + IAC | **62.64 (+3.31)** | **61.20 (+1.39)** | **64.64 (+1.67)** |

**PF 扩展到 RNN 方法 (ExpRate %)**

| 方法 | CROHME 2014 | CROHME 2016 | CROHME 2019 |
|------|-------------|-------------|-------------|
| DWAP | 50.10 | 47.50 | - |
| DWAP + PF | **57.10 (+7.00)** | **56.23 (+8.73)** | **57.30** |
| CAN | 55.27 | 53.97 | 52.96 |
| CAN + PF | **57.30 (+2.03)** | **56.84 (+2.87)** | **57.13 (+4.17)** |
| ABM | 56.85 | 52.92 | 53.96 |
| ABM + PF | **58.11 (+1.26)** | **56.50 (+3.58)** | **54.30 (+0.34)** |

### 关键发现

1. **位置森林（PF）贡献最大**：单独加入 PF 在 CROHME 2014 上提升 2.80%，IAC 进一步贡献 0.51%
2. **PF 是通用插件**：扩展到 DWAP、ABM、CAN 等 RNN 方法后，均获得显著提升（最高 +8.73%），证明了方法的广泛适用性
3. **在容错指标上提升更显著**：≤1 和 ≤2 错误的提升比 ExpRate 更明显，说明 PF 有效增强了模型对复杂公式的纠错能力
4. **推理零开销**：位置森林和位置识别头仅在训练时使用，推理时完全移除，不增加任何延迟或计算成本
5. **在复杂表达式上优势最大**：在构造的多层嵌套 MNE 数据集上，PosFormer 提升高达 4.62%，验证了显式位置建模对复杂表达式的重要性

## 亮点与洞察

1. **"免费午餐"设计哲学**：训练时通过辅助任务获得更好的特征，推理时完全不增加开销——这是非常优雅的设计
2. **从 LaTeX 语法规则自动生成监督信号**：不需要额外标注，仅从已有的 LaTeX 序列中解析出位置信息作为辅助标签
3. **对 CoMER 注意力校正的精准诊断和修复**：明确指出结构符号 vs 实体符号的注意力语义差异，修复方案简洁有效（仅需引入一个指示函数）
4. **森林 vs 树的选择**：使用森林结构而非单棵树，自然处理数学表达式中独立子结构的并列关系，比树结构更灵活

## 局限与展望

1. **最大嵌套层级固定为 3**：对于极深嵌套的表达式可能不够，虽然实际中超过 3 层嵌套较少见
2. **未引入语言模型**：论文比较了语言感知方法 RLFN，PosFormer 作为无语言模型方法仍有 6.25% 优势，但结合语言模型可能进一步提升
3. **DenseNet 骨干相对陈旧**：可以尝试更强的视觉骨干（如 Swin Transformer）或预训练视觉编码器
4. **位置森林的构建依赖 LaTeX 语法假设**：对于非标准 LaTeX 写法或新类型结构可能需要扩展编码规则
5. **MNE 数据集规模较小**：作者自建的复杂表达式测试集可以进一步扩大以提供更充分的评估

## 相关工作与启发

- **与 CoMER 的关系**：PosFormer 以 CoMER 为基线，在其注意力校正机制上做了精准改进（IAC），并额外引入位置森林辅助任务
- **与树结构方法的关系**：PosFormer 不直接用树结构解码，而是将位置信息编码为森林结构作为辅助监督，结合了树方法和序列方法的优点
- **多任务学习启发**：通过设计与主任务强相关的辅助任务来改善特征表示质量，这一思路可推广到其他结构化序列预测任务（如化学公式识别、乐谱识别等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 位置森林编码和零注意力校正都是新颖且合理的设计，尤其是"训练时用、推理时移除"的辅助任务范式
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖单行/多行/复杂/大规模四类数据集，消融彻底，还验证了在三种 RNN 基线上的可扩展性
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，算法伪代码和图示有助于理解，动机阐述充分
- **价值**: ⭐⭐⭐⭐ — 显著推进了 HMER 的 SOTA，且 PF 作为即插即用模块具有良好的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](../../ICML2025/self_supervised/pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[CVPR 2025\] MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](../../CVPR2025/self_supervised/metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](../../ICML2026/self_supervised/flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)

</div>

<!-- RELATED:END -->
