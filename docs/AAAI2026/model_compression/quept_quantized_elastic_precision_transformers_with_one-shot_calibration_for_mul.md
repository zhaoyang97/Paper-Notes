---
title: >-
  [论文解读] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching
description: >-
  [AAAI 2026][模型压缩][弹性精度量化] 提出QuEPT弹性精度量化框架，通过Multi-Bit Token Merging和Multi-Bit Cascaded LoRA两大核心模块，实现一次校准即可在ViT/LLM/MLLM上实时切换任意预定义位宽，性能媲美甚至超越单位宽SOTA PTQ方法。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "弹性精度量化"
  - "后训练量化"
  - "多位宽切换"
  - "级联LoRA"
  - "Token合并"
---

# QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching

**会议**: AAAI 2026  
**arXiv**: [2602.12609](https://arxiv.org/abs/2602.12609)  
**代码**: [https://github.com/xuke225/QuEPT](https://github.com/xuke225/QuEPT)  
**领域**: 模型压缩  
**关键词**: 弹性精度量化, 后训练量化, 多位宽切换, 级联LoRA, Token合并

## 一句话总结
提出QuEPT弹性精度量化框架，通过Multi-Bit Token Merging和Multi-Bit Cascaded LoRA两大核心模块，实现一次校准即可在ViT/LLM/MLLM上实时切换任意预定义位宽，性能媲美甚至超越单位宽SOTA PTQ方法。

## 研究背景与动机

模型量化通过低位定点数表示权重和激活来降低计算和存储开销。然而，现有大多数量化方法只能实现单一预定义位宽，修改位宽需要重新优化。**弹性精度量化**(Elastic Precision Quantization)可以通过一次优化同时处理多个预定义位宽，使量化模型动态适应不同位宽需求。

**从CNN到Transformer的挑战**：多位宽量化在CNN上已有较好成果（RobustQuant、Any-Precision、EQ-Net等），但Transformer有其独特难点：

**密集的token间依赖**：注意力机制带来的动态稀疏性使量化噪声放大

**异质的层级位宽敏感性**：不同层对不同位宽的敏感度差异极大

**激活范围大**：Transformer的激活分布范围远超CNN

**现有多位宽量化方法的局限**：
- **Any-Precision LLM**：仅做权重量化，不考虑激活量化，低精度性能差
- **MatQuant**：使用co-training和co-distillation正则化，但没考虑不同位宽之间的竞争关系
- **基于QAT的方法**（RobustQuant、EQ-Net等）：训练开销过大，不适用于大规模Transformer

**核心矛盾**：在多位宽联合优化时，低位宽的激进量化会负面影响中高位宽的表示质量，导致整体性能受限于最低精度配置。不同位宽之间存在"竞争冲突"。

**切入角度**：设计PTQ框架，通过Token合并策略维持跨位宽平衡性能，通过级联LoRA结构建立位宽间的层次化参数共享，将冲突转化为协同。

## 方法详解

### 整体框架

QuEPT的Pipeline为逐block重建：
1. 初始化权重裁剪参数和级联LoRA适配器（权重和量化尺度冻结）
2. 将目标位宽集 $\mathcal{B}$ 分为低位组 $\mathcal{B}_L$、中位组 $\mathcal{B}_M$ 和高位组 $\mathcal{B}_H$
3. 每步从三个组各采样一个位宽，通过MB-ToMe融合前一block的多位宽特征
4. 当前block在三个采样位宽下前向传播，重建损失反向传播更新级联LoRA和裁剪阈值
5. 部署时只需选择对应的LoRA切片即可切换到任意位宽

### 关键设计

#### 1. **Multi-Bit Token Merging (MB-ToMe)**
- **功能**：在多位宽优化过程中融合不同位宽的token特征，维持跨精度的平衡性能
- **核心思路**：探索了三种策略：
    - Case 1（随机选择）：每个token随机选一个位宽的特征 → 质量不稳定
    - Case 2（均匀融合）：三个位宽组1:1:1加权融合 → 丢失高位宽细节
    - Case 3（选择性合并，最终方案）：基于量化鲁棒性选择性保留+融合
- **关键公式**：
$$X_k' = \begin{cases} X_k^H, & \text{if } k \in \Phi \\ \lambda_1 X_k^H + \lambda_2 X_k^M + \lambda_3 X_k^L, & \text{else} \end{cases}$$
  其中 $\Phi$ 是通过8-bit和4-bit特征余弦相似度排序选出的Top-p%稳定token的索引集合
- **设计动机**：鲁棒token的高精度特征应被保留作为结构支撑，不稳定token通过融合维持特征连续性。合并不相似的token会导致更均匀的数值分布，对不同位宽更鲁棒

#### 2. **Multi-Bit Cascaded LoRA (MB-CLoRA)**
- **功能**：通过层次化的LoRA参数共享结构，建立不同位宽之间的协同关系
- **核心思路**：所有位宽共享统一的LoRA参数 $A \in \mathbb{R}^{r \times q}$ 和 $B \in \mathbb{R}^{p \times r}$，通过级联截取不同rank:
$$R^{(b)} = B_{[:,:r_b]} A_{[:r_b,:]}$$
  rank分配遵循级联模式：低位宽用更多rank
$$r_b = \begin{cases} r_h & b \in \mathcal{B}_H \\ r_h + r_m & b \in \mathcal{B}_M \\ r_h + r_m + r_l & b \in \mathcal{B}_L \end{cases}$$
- **设计动机**：低位宽量化误差更大，需要更多补偿能力。级联结构使低位宽的补偿矩阵自然包含高位宽的参数作为前导子矩阵，建立继承关系。高精度梯度可以自然扩散到低精度对应项

#### 3. **权重裁剪参数联合优化**
- **功能**：与LoRA参数同时优化权重裁剪阈值
- **核心公式**：
$$s_w^b = \frac{\alpha_b \times \max(W + B_b A_b) - \beta_b \times \min(W + B_b A_b)}{2^{b-1}}$$
- **设计动机**：LoRA参数更新过程中量化权重持续变化，固定裁剪阈值不再最优。裁剪机制处理权重异常值造成的大幅误差，为LoRA创造更平滑的误差信号

### 损失函数 / 训练策略

- 使用MAE损失（实验证明优于MSE）进行逐block重建：
$$\min_{A,B,\alpha,\beta} \sum_{b \in \{b_L, b_M, b_H\}} \|WX - \widehat{(W+R^{(b)})} X'\|_1$$
- ViT校准数据：1024张ImageNet无标签图像
- LLM校准数据：128条C4样本
- MLLM校准数据：128对ShareGPT4V图文对
- LoRA放在裁剪算子内部，推理时无额外开销
- 可通过KL散度评估各层敏感度，结合DP算法实现混合精度

## 实验关键数据

### 主实验（ViT ImageNet Top-1精度）

| 模型 | 方法 | 类型 | W4A4 | W5A5 | W6A6 | W8A8 | 时间 |
|------|------|------|------|------|------|------|------|
| ViT-S | ERQ | 单位宽 | 68.9 | 78.8 | 80.5 | 81.2 | 9×N |
| ViT-S | PTMQ | 多位宽 | - | - | 76.1 | 78.2 | 430min |
| ViT-S | **QuEPT** | 多位宽 | **75.1** | **79.7** | **80.6** | **81.2** | **17min** |
| ViT-B | PTMQ | 多位宽 | - | - | 77.7 | 79.1 | 950min |
| ViT-B | **QuEPT** | 多位宽 | **80.7** | **83.3** | **83.8** | **84.3** | **36min** |

QuEPT在ViT-S上W4A4比ERQ高6.2%，训练时间仅为PTMQ的1/26。

### LLM实验（LLaMA系列）

| 模型 | 位宽 | 方法 | WikiText2 PPL↓ | C4 PPL↓ | 5任务平均↑ |
|------|------|------|---------------|---------|-----------|
| L2-7B | W4A4 | DuQuant | 6.28 | 7.90 | 59.11 |
| L2-7B | W4A4 | **QuEPT** | **6.33** | **7.86** | **61.62** |
| L3-8B | W4A4 | DuQuant | 8.56 | 11.98 | 65.05 |
| L3-8B | W4A4 | **QuEPT** | **8.25** | **11.67** | **67.04** |

### MLLM实验（LLaVA-OneVision-7B）

| 位宽 | 方法 | MMMU | OCRBench | TextVQA | VizWiz | SEED |
|------|------|------|----------|---------|--------|------|
| W3A16 | MBQ | 42.0 | 61.1 | 73.3 | 60.7 | 66.4 |
| W3A16 | **QuEPT** | **44.6** | 60.6 | **74.1** | 60.3 | **71.6** |
| W4A8 | MBQ | 42.6 | 52.3 | 68.3 | 58.9 | 64.4 |
| W4A8 | **QuEPT** | **43.4** | **61.2** | **71.5** | **61.3** | **70.7** |

### 消融实验

**MB-CLoRA策略对比（LLaMA2-7B平均精度）：**

| 策略 | W4A4 | W5A5 | W8A8 |
|------|------|------|------|
| 完全共享(Case 1) | 60.9 | 64.2 | 65.5 |
| 独立LoRA(Case 2) | 59.2 | 64.4 | 65.7 |
| **MB-CLoRA(Case 3)** | **61.6** | **64.5** | 65.6 |

**MB-ToMe策略对比（LLaMA2-7B）：**

| 策略 | W4A4 | W6A6 | W8A8 |
|------|------|------|------|
| 随机选择(Case 1) | 55.7 | 64.7 | 65.4 |
| 均匀融合(Case 2) | 60.2 | 65.2 | 65.5 |
| **选择性合并(Case 3)** | **61.6** | **65.5** | **65.6** |

### 关键发现

1. MB-ToMe在低位宽(W4A4)下增益最大（比Case 1高5.9%），因为低位宽时token特征最不稳定，选择性保留高精度特征至关重要
2. MB-CLoRA的级联共享在低位宽性能上优于完全独立和完全共享策略，因为它让低位宽"继承"高位宽的优化成果
3. 使用MAE损失优于MSE损失
4. 混合精度实验中，平均位宽2.25/3.00/4.00时WikiText2 PPL仅为8.97/5.93/5.54

## 亮点与洞察

1. **统一多模态框架**：首个在ViT、LLM和MLLM上统一验证的多位宽PTQ方法
2. **将冲突转化为协同**：MB-ToMe和MB-CLoRA的组合从不同角度解决位宽间竞争问题——前者在特征空间缓解，后者在参数空间协同
3. **LoRA的巧妙放置**：将LoRA放入裁剪算子内部，确保推理时无额外开销，部署友好
4. **弹性到混合精度的无缝转换**：通过层敏感度+DP算法即可将弹性量化转为混合精度，无需重新训练
5. **训练效率极高**：ViT-S仅17分钟，是PTMQ的1/26

## 局限与展望

1. 未显式处理LLM中的异常值(outlier)，与SpinQuant等outlier缓解技术结合可能进一步提升
2. 极低位宽(如W2)性能仍有限
3. 校准数据量小(128条)，更大校准集可能带来提升但会增加时间
4. MB-ToMe中的超参数 $\lambda_1, \lambda_2, \lambda_3$ 和top-p%需要调节

## 相关工作与启发

- **Any-Precision LLM (Park et al., 2024)**：基于截断位宽和增量上采样，但不支持激活量化
- **MatQuant (Nair et al., 2025)**：co-distillation正则化，未考虑位宽间竞争
- **SVDQuant (Li et al., 2025b)**：让LoRA学习outlier分支使剩余权重更易量化
- **QuaRot (Ashkboos et al., 2024)**：旋转矩阵消除outlier的4-bit推理
- **PTMQ (Xu et al., 2024a)**：CNN/ViT上的多位宽PTQ，但计算开销大

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[AAAI 2026\] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)
- [\[AAAI 2026\] First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](first-order_error_matters_accurate_compensation_for_quantized_large_language_mod.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](../../ICML2026/model_compression/multi-adapter_representation_interventions_via_energy_calibration.md)

</div>

<!-- RELATED:END -->
