---
title: >-
  [论文解读] MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation
description: >-
  [AAAI 2026][视频生成][多主体视频生成] 提出 MoFu，通过 Scale-Aware Modulation（LLM 引导的尺度感知调制）和 Fourier Fusion（基于 FFT 的排列不变特征融合）两个核心模块，同时解决多主体视频生成中的**尺度不一致**和**排列敏感性**两大挑战，并构建了 MoFu-1M 训练数据集和 MoFu-Bench 评测基准。
tags:
  - "AAAI 2026"
  - "视频生成"
  - "多主体视频生成"
  - "尺度一致性"
  - "排列不变性"
  - "傅里叶融合"
  - "DiT"
---

# MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation

**会议**: AAAI 2026  
**arXiv**: [2512.22310](https://arxiv.org/abs/2512.22310)  
**代码**: 无  
**领域**: 视频理解 / 视频生成  
**关键词**: 多主体视频生成, 尺度一致性, 排列不变性, 傅里叶融合, DiT

## 一句话总结

提出 MoFu，通过 Scale-Aware Modulation（LLM 引导的尺度感知调制）和 Fourier Fusion（基于 FFT 的排列不变特征融合）两个核心模块，同时解决多主体视频生成中的**尺度不一致**和**排列敏感性**两大挑战，并构建了 MoFu-1M 训练数据集和 MoFu-Bench 评测基准。

## 研究背景与动机

### 领域现状

多主体视频生成目标是从文本提示和多张参考图像合成视频，使每个主体保持视觉保真度和自然尺度。随着 Wan2.1、HunyuanVideo、Sora 等大型视频生成模型的进展，视频生成质量不断提升。近期 Phantom、SkyReels-A2、VACE 等方法开始尝试多主体生成，但仍面临根本性挑战。

### 现有痛点

**尺度不一致（Scale Inconsistency）**：参考图像中主体的缩放级别不同——比如一只小鸟的特写和一头大象的远景——导致生成视频中主体出现不自然的大小，如碗比桌子还大，或麻雀和大象一样大

**排列敏感性（Permutation Sensitivity）**：现有方法按顺序处理参考图像（沿帧或通道维度逐一插入），**改变输入顺序会导致主体变形、消失或物理上不合理的交互**。而且随着参考图像数量增加，计算成本急剧上升

### 现有尝试的局限

- **解决尺度问题**：Phantom、SkyReels 尝试将文本提示融合到参考图像表示中，隐式利用提示中编码的空间关系（如"女孩怀里抱着一只狗"）。但**缺乏显式机制**来解读和强制自然尺度
- **解决排列问题**：CustomVideo、MAGREF 将多张参考图像拼接到单个画布上形成统一输入。但这引入了**空间偏差**——中心区域或较大的主体可能获得不成比例的注意力

### 核心矛盾

多主体生成需要**同时保证尺度合理性和输入顺序无关性**。尺度问题的根源是参考图像的缩放差异没有被显式考虑；排列问题的根源是顺序处理或空间拼接不可避免地引入了位置偏差。两个问题相互交织但需要不同的技术解决方案。

### 本文切入角度

设计两个正交但互补的模块：用 LLM 提取文本中的隐式尺度线索来**显式调制**特征（解决尺度），用 FFT 将参考特征变换到频域后**对称求和**实现排列不变融合（解决排列）。

## 方法详解

### 整体框架

MoFu 基于 DiT（Diffusion Transformer）骨干网络，集成三个组件：
1. **Scale-Aware Modulation (SMO)**：尺度感知调制模块
2. **Fourier Fusion**：傅里叶融合策略
3. **Scale-Permutation Stability Loss (SPSL)**：尺度-排列稳定性损失

### 关键设计

#### 1. Scale-Aware Modulation (SMO) — 尺度感知调制

**功能**：从文本提示中提取主体间的相对尺度关系，并注入到生成过程中。

**核心思路**：

**Step 1：LLM 提取尺度线索**
- 使用冻结的 LLM（Qwen2.5）编码文本提示：$\mathbf{e}_p = \text{LLM}(p) \in \mathbb{R}^d$
- 提取的不是文本输出，而是 [CLS] 嵌入作为尺度感知表示
- 通过精心设计的 prompt 让 LLM 推断场景中实体间的成对相对大小关系

**Step 2：Scale Control Adapter (SCA) 预测调制参数**
- 轻量 MLP 将语义嵌入映射为调制三元组：$\gamma, \beta, \eta = f_{\text{SCA}}(\mathbf{e}_p)$
- $\gamma, \beta$：缩放和偏移因子
- $\eta$：门控因子控制残差连接

**Step 3：自适应特征调制**

$$\hat{\mathbf{F}} = \gamma \odot \mathbf{F} + \beta, \quad \mathbf{F}_{out} = \mathbf{F} + \eta \cdot \text{Layer}(\hat{\mathbf{F}})$$

其中 Layer 是 MHA 或 FFN 模块。

**设计动机**：
- 文本提示是尺度信息的天然来源——"一个女孩怀里抱着一只小猫"隐含了猫远小于人
- 利用 LLM 的常识推理能力理解这些隐式尺度关系
- 通过 DiT 原有的调制机制注入，不破坏模型架构

#### 2. Fourier Fusion — 傅里叶融合

**功能**：将多张参考图像融合为排列不变的统一表示。

**核心思路**：

**数学基础**：高维概率中的定理——"高维随机向量间的内积趋于零，即近似正交"。这意味着在频域中直接求和不会导致严重的信息干扰。

**具体步骤**：

**Step 1：参考图像预处理**
- 使用 Grounded-SAM 分割每张参考图像，裁剪主体区域
- 通过 3×3 CNN 编码器提取特征图：$\mathbf{F}_i = \mathcal{E}(x_i) \in \mathbb{R}^{d \times H \times W}$

**Step 2：FFT 变换与频域分解**
$$\mathcal{F}_i = \text{FFT}(\mathbf{F}_i)$$
$$\mathcal{F}_i^{\text{HF}} = M_{\text{freq}} \odot \mathcal{F}_i, \quad \mathcal{F}_i^{\text{LF}} = (1 - M_{\text{freq}}) \odot \mathcal{F}_i$$

**Step 3：利用近似正交性直接求和**
$$\mathcal{F}^{\text{HF}} = \sum_{i=1}^{N} \mathcal{F}_i^{\text{HF}}, \quad \mathcal{F}^{\text{LF}} = \sum_{i=1}^{N} \mathcal{F}_i^{\text{LF}}$$

**Step 4：IFFT 重建**
$$\mathbf{F}_{\text{fused}} = \text{IFFT}(\mathcal{F}^{\text{HF}} + \mathcal{F}^{\text{LF}})$$

融合后的表示与视频特征拼接，作为生成条件。

**设计动机**：
- 加法运算天然满足**交换律**，无论输入顺序如何，求和结果相同 → 排列不变
- 频域分解保留高频（纹理细节）和低频（整体结构）的区分
- 避免了顺序拼接的位置偏差和画布拼接的空间偏差
- 计算开销极低：仅需 FFT + 逐元素求和 + IFFT

#### 3. Scale-Permutation Stability Loss (SPSL)

**功能**：联合监督尺度一致性和排列不变性。

**尺度损失**（Scale Loss）：
- 根据参考 mask 的面积比自适应加权标准 MSE 损失：

$$w_r = \frac{\exp(a_r)}{\sum_{r'}\exp(a_{r'})}, \quad \mathbf{M} = \sum_{r=1}^{R} w_r \cdot \text{Resize}(m_r)$$

$$\mathcal{L}_{\text{scale}} = \frac{\sum(\mathcal{L}_{\text{mse}} \odot \mathbf{M})}{\sum \mathbf{M} + \epsilon}$$

面积比大的主体获得更大的损失权重。

**排列损失**（Permutation Loss）：
- 在频域计算 $P$ 种排列的 MSE 损失：

$$\mathcal{L}_{\text{perm}} = \frac{1}{P} \sum_{p=1}^{P} \|F(\mathcal{R}_p) - F(\mathcal{R}_{\text{ref}})\|_2^2$$

**最终损失**：$\mathcal{L}_{\text{SPSL}} = \mathcal{L}_{\text{scale}} + \mathcal{L}_{\text{perm}}$

**设计动机**：SMO 和 Fourier Fusion 分别需要尺度信息和排列信息的监督，SPSL 提供了这些监督信号。没有 SPSL，这两个模块无法有效学习。

### 训练策略

- **数据集 MoFu-1M**：从 15M 片段经两阶段过滤得到 2.5M 高质量片段，选择 1M 作为最终训练集
- AdamW 优化器，β₁=0.9，β₂=0.999，权重衰减 0.01
- 初始学习率 1e-5，余弦退火+周期重启
- 16 × NVIDIA H800 训练 7 天
- 输入分辨率 480P，81 帧

## 实验关键数据

### 主实验

| 方法 | Aesthetics↑ | FaceSim↑ | GmeScore↑ | Motion↔ | ScaleScore↑ | SubjectSim↑ |
|------|-----------|----------|-----------|---------|------------|------------|
| Phantom | 0.355 | 0.375 | 0.706 | 0.229 | 0.536 | 0.748 |
| SkyReels-A2 | 0.286 | 0.341 | 0.691 | 0.233 | 0.527 | 0.737 |
| VACE | 0.392 | 0.247 | 0.732 | 0.214 | 0.547 | 0.692 |
| MAGREF | 0.369 | 0.362 | 0.717 | 0.207 | 0.511 | 0.731 |
| **MoFu** | **0.401** | **0.396** | **0.745** | 0.221 | **0.585** | **0.755** |

- ScaleScore 提升最大：0.585 vs 第二名 VACE 的 0.547（+7.0%）
- 几乎所有指标上全面领先

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无 SMO | 麻雀与大象一样大 | 尺度关系完全丢失 |
| 有 SMO | 自然的体型比例保持 | LLM 尺度线索有效 |
| 无 Fourier Fusion | 改变参考图像顺序导致木箱消失 | 排列敏感 |
| 有 Fourier Fusion | 所有主体无论顺序均完整生成 | 排列不变 |

注：作者指出 SPSL 不提供独立消融的原因是——SMO 和 Fourier Fusion 分别依赖 SPSL 的尺度和排列监督信号，去掉 SPSL 而保留它们没有意义。

### MoFu-Bench 统计

- 1000 个主体-文本对
- 参考图像类别分布：人物 26%、动物 17%、服装 10%、物体 30%、卡通 4%、其他 13%
- Prompt 长度主要在 100-200 词
- 每个案例最多 3 张参考图像，带系统性尺度变化和随机排列

### 关键发现

1. **尺度问题不仅是视觉问题，更是语义理解问题**：需要理解"大象远大于麻雀"这样的常识
2. **频域求和的理论支撑有力**：高维近似正交性使简单求和成为有效的排列不变操作
3. **接尺度损失中用 softmax 归一化面积比**作为权重，巧妙地将空间先验转化为损失权重
4. **I2V 方法（如 MAGREF）在 FaceSim 上表现好**但尺度一致性差，说明参考条件化与尺度推理是不同的能力

## 亮点与洞察

1. **理论与实践的优美结合**：用高维概率的近似正交性定理为频域求和提供理论支撑
2. **LLM 作为尺度推理器**：创新地利用 LLM 的常识知识理解文本中隐含的尺度关系
3. **两个问题、两个模块、一个损失**：设计的对称性和互补性体现了良好的系统设计
4. **MoFu-Bench 填补空白**：首个专门评估尺度一致性和排列不变性的多主体视频基准
5. **工程细节扎实**：数据构建从 15M 到 1M 的多阶段过滤，评估指标覆盖 6 个维度

## 局限与展望

1. **强调空间推理可能限制时间建模**：高度动态场景中的精细时序交互可能表现不佳
2. **依赖高质量参考输入**：参考图像稀疏或噪声大时性能可能下降
3. **统一架构的通用性代价**：缺乏领域特定先验（如人脸特定优化）
4. **ScaleScore 依赖 GPT-4o 判断**：评估指标本身的可靠性取决于 GPT-4o 的尺度判断能力
5. **主体数量限制**：目前演示最多 3 个主体，更多主体的情况待验证
6. **训练成本高**：16×H800 训练 7 天，资源要求不低

## 相关工作与启发

- **FFT 在表示学习中的应用**：频域操作（特别是傅里叶变换）作为实现特定数学性质（如排列不变性）的工具，值得在其他场景探索
- **DiT 调制机制的灵活性**：DiT 的 $\gamma \cdot \text{LayerNorm}(h) + \beta$ 调制机制可以方便地注入各种条件信号
- **LLM 作为中间推理器**：不是用 LLM 的文本输出，而是提取其嵌入用于下游任务——这种用法在多模态系统中越来越常见
- **数据构建的系统性方法**：场景检测→美学过滤→文本对齐→主体提取→面部处理的多阶段管线具有可复用性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 傅里叶融合+LLM尺度感知调制的组合很有创意，理论支撑充分
- **实验充分度**: ⭐⭐⭐⭐ — 定量和定性全面，但消融中 SPSL 未独立消融
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法阐述系统
- **价值**: ⭐⭐⭐⭐ — 首次系统性解决多主体生成的尺度和排列问题，基准和方法都有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ID-Crafter: VLM-Grounded Online RL for Compositional Multi-Subject Video Generation](../../CVPR2026/video_generation/id-crafter_vlm-grounded_online_rl_for_compositional_multi-subject_video_generati.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](../../CVPR2026/video_generation/facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2025\] Multi-subject Open-set Personalization in Video Generation](../../CVPR2025/video_generation/multi-subject_open-set_personalization_in_video_generation.md)
- [\[CVPR 2026\] UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation](../../CVPR2026/video_generation/unityvideo_unified_multi-modal_multi-task_learning_for_enhancing_world-aware_vid.md)
- [\[CVPR 2026\] CineBrain: A Large-Scale Multi-Modal Audiovisual Brain Dataset for Brain-Conditioned Video Generation](../../CVPR2026/video_generation/cinebrain_a_large-scale_multi-modal_audiovisual_brain_dataset_for_brain-conditio.md)

</div>

<!-- RELATED:END -->
