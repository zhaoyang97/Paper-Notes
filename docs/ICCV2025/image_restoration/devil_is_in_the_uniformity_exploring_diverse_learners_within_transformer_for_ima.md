---
title: >-
  [论文解读] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration
description: >-
  [ICCV 2025][图像恢复][注意力机制] 针对标准Multi-Head Attention (MHA)中各head使用均匀子空间导致的冗余问题，提出HINT模型，通过异构层级多头注意力(HMHA)和Query-Key缓存更新(QKCU)机制增强head间多样性与交互，在5类图像恢复任务的12个benchmark上取得SOTA结果。
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "注意力机制"
  - "Transformer"
  - "低光增强"
  - "去雾去雪"
---

# Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration

**会议**: ICCV 2025  
**arXiv**: [2503.20174](https://arxiv.org/abs/2503.20174)  
**代码**: [https://github.com/joshyZhou/HINT](https://github.com/joshyZhou/HINT)  
**领域**: 图像恢复  
**关键词**: 图像恢复, Multi-Head Attention, Transformer, 低光增强, 去雾去雪

## 一句话总结
针对标准Multi-Head Attention (MHA)中各head使用均匀子空间导致的冗余问题，提出HINT模型，通过异构层级多头注意力(HMHA)和Query-Key缓存更新(QKCU)机制增强head间多样性与交互，在5类图像恢复任务的12个benchmark上取得SOTA结果。

## 研究背景与动机
- **领域现状**：Transformer在图像恢复中取得显著成功，MHA是其核心组件，通过多个head并行执行注意力计算以捕获多样化特征
- **现有痛点**：标准MHA为每个head分配相同大小(C/h)的子空间，导致不同head倾向于关注相同区域，产生冗余——NLP领域已证明只有少数head对最终决策起关键作用，其余可被剪枝
- **核心矛盾**：均匀子空间分割使head包含相似信息 + head之间缺乏协作 → 两方面共同加剧冗余问题
- **切入角度**：从两个维度改进MHA——(1) 让head从不同大小的子空间学习多样化特征；(2) 引入层内和跨层的head交互机制
- **核心idea**：通过通道相似性重排+层级子空间切分实现多样化学习，通过Query-Key缓存在层内和跨层传递信息增强head协作

## 方法详解

### 整体框架
HINT采用编码器-解码器架构，共N1=4级。输入退化图像经卷积层提取浅层特征后，通过N1级恢复管线生成深层特征。编码器仅含FFN（不对称设计），解码器包含HMHA + FFN。编码器-解码器间有跳跃连接。之后是N3=4块的细化阶段(refinement stage)，最终输出残差图像。

### 关键设计
1. **Hierarchical Multi-Head Attention (HMHA)**:

    - 功能：让不同head在不同大小的子空间中学习，捕获多样化的上下文信息
    - 核心思路：将通道空间按 $C = [C_1, C_2, ..., C_h]$（$C_1 \leq C_2 \leq ... \leq C_h$）进行层级切分。切分前先基于Pearson相关性对通道进行相似性重排(reranking)，确保每个子空间包含语义独立的信息。维度比例设为[1, 2, 2, 3]，head数4
    - 设计动机：均匀分割使子空间包含相似信息→head关注相同区域。通过差异化子空间大小+重排，迫使不同head学习层次化、互补的表示

2. **Query-Key Cache Updating (QKCU) - 层内调制**:

    - 功能：增强同一层内各head之间的信息交互
    - 核心思路：维护一个IntraCache，将Q+K的拼接与HMHA输出相加后，经门控机制和压缩-重建变换：
      $\mathbf{F}_{gated} = \text{GELU}(\text{Conv}(\mathbf{F}_{intra}^s)) \odot \mathbf{F}_{intra}^s$
      $\mathbf{F}_{Intra}^o = \text{Conv}_{up}(\text{Conv}_{down}(\mathbf{F}_{gated}))$
    - 设计动机：门控机制选择性保留最有信息量的元素，压缩-重建迫使模型聚焦关键特征

3. **Query-Key Cache Updating (QKCU) - 跨层调制**:

    - 功能：利用历史注意力分数调制当前层的注意力计算
    - 核心思路：维护InterCache存储历史QK^T注意力矩阵。对当前层注意力分数进行scale-shift调制：
      $\mathbf{F}_m = \mathbf{F}_{scale} \odot \mathbf{F}_{att} + \mathbf{F}_{shift}$
      跨层缓存渐进更新：$\mathbf{F}_{inter} = \alpha \mathbf{F}_{inter} + (1-\alpha) \mathbf{F}_{inter}^l$，α=0.9
    - 设计动机：传统MHA中同一网络不同层的head也独立工作，跨层缓存让模型利用历史注意模式来引导当前层的注意力分配，实现动态（依赖输入的）调制

### 损失函数 / 训练策略
- 使用AdamW优化器
- 嵌入维度C=48，4个head，维度比[1, 2, 2, 3]
- 编码器-解码器块数：[4, 6, 6, 6]，第4级为瓶颈层
- 细化阶段4个块
- α=0.9控制跨层缓存信息流动

## 实验关键数据

### 主实验 - 低光增强 (LOL-v2)

| 方法 | LOL-v2-real PSNR | LOL-v2-syn PSNR | 平均 PSNR | 平均 SSIM |
|------|-----------------|-----------------|----------|----------|
| Restormer (CVPR'22) | 19.94 | 21.41 | 20.68 | 0.829 |
| MambaIR (ECCV'24) | 21.25 | 25.55 | 23.40 | 0.880 |
| Retinexformer (ICCV'23) | 22.80 | 25.67 | 24.24 | 0.885 |
| MambaLLIE (NeurIPS'24) | 22.95 | 25.87 | 24.41 | 0.894 |
| **HINT** | **23.11** | **27.17** | **25.14** | **0.917** |

HINT比Retinexformer高0.9dB，比通用恢复方案至少高1.74dB。

### 其他任务主结果

**去雪 (Snow100K)**:

| 方法 | PSNR | SSIM |
|------|------|------|
| AST (CVPR'24) | 32.50 | 0.96 |
| ConvIR-S (TPAMI'24) | 33.79 | 0.95 |
| **HINT** | **34.14** | 0.94 |

**去雾 (SOTS)** - 多方法比较：

| 方法 | PSNR | SSIM |
|------|------|------|
| PromptIR (NeurIPS'23) | 31.31 | 0.973 |
| AdaIR (ICLR'25) | 31.80 | 0.981 |
| **HINT** | **32.24** | **0.981** |

### 消融实验

**注意力机制消融** (LOL-v2-syn):

| 配置 | PSNR | SSIM |
|------|------|------|
| W-MSA [Uformer] | 24.19 | 0.941 |
| MDTA [Restormer] | 26.42 | 0.948 |
| **HMHA (本文)** | **27.17** | **0.950** |

**HMHA重排策略消融**:

| 配置 | 参数量(M) | PSNR | SSIM |
|------|----------|------|------|
| 无重排 [Restormer] | 24.76 | 26.42 | 0.948 |
| 随机打乱 | 24.87 | 26.54 | 0.949 |
| **HMHA (Pearson重排)** | **24.87** | **27.17** | **0.950** |

**QKCU模块消融**:

| IntraCache | InterCache | 参数量(M) | PSNR |
|------------|------------|----------|------|
| ✗ | ✗ | 21.34 | 26.47 |
| ✓ | ✗ | 23.82 | 26.67 |
| ✗ | ✓ | 22.39 | 26.72 |
| **✓** | **✓** | **24.87** | **27.17** |

### 关键发现
- HMHA比W-MSA提升2.98dB，比MDTA提升0.75dB，验证了层级子空间的有效性
- Pearson相关性重排比随机打乱高0.63dB，比不重排高0.75dB——重排策略对性能提升至关重要
- QKCU的层内和跨层模块各提供约0.2-0.25dB提升，合计0.7dB，仅增加16.5%参数
- 特征可视化清晰显示MDTA的head关注相同区域，而HMHA的head关注不同区域
- 在真实场景数据集(DICM/MEF/NPE/VV，无GT)上MANIQA指标也领先
- 模型效率：126.92G FLOPs + 24.87M参数，与Restormer(144.25G+26.13M)相当

## 亮点与洞察
- 问题诊断精准：通过可视化直接展示标准MHA中head的冗余——不同head关注相同区域（红框）而遗漏退化区域（黄框）
- 层级子空间切分+通道重排的组合简洁有效，几乎不增加计算成本却带来显著性能提升
- 跨层QK缓存的指数移动平均更新(α=0.9)实现了渐进式历史信息融合
- 跨5类任务(低光增强/去雾/去雪/去噪/去雨)12个benchmark的全面评估展示了方法的通用性

## 局限与展望
- 维度比[1,2,2,3]和α=0.9为手动设定，自适应确定这些超参数可能进一步提升效果
- Pearson相关性重排为静态（基于通道全局统计），动态的、依赖输入的重排可能更优
- 编码器不使用注意力（不对称设计）会限制编码端的全局建模能力
- 缺少与最新Mamba架构(如MambaIR)在更多任务上的系统性对比
- 跨层InterCache在模型深度极大时的存储与计算开销未充分讨论

## 相关工作与启发
- Restormer的MDTA(沿通道维度计算注意力)是关键baseline，HINT在其基础上改进MHA内部机制
- head冗余问题在NLP(head pruning)中已有研究，本文首次系统性地将其迁移到图像恢复领域
- QKCU的scale-shift调制思路与FiLM条件化方法相通，但结合了QK注意力信息的历史缓存
- 层级子空间切分思想可推广到其他Transformer架构（如目标检测、分割等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 层级子空间切分+QK缓存交互的组合有新意，但各单独组件的思路有前人工作基础
- 实验充分度: ⭐⭐⭐⭐⭐ 12个benchmark、5类任务、全面消融、效率分析、真实场景评估、下游任务评估
- 写作质量: ⭐⭐⭐⭐ 问题诊断+可视化支撑清晰有力，方法描述详细
- 价值: ⭐⭐⭐⭐ 方法通用性强，可作为即插即用模块改进现有Transformer恢复模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[CVPR 2025\] Complexity Experts are Task-Discriminative Learners for Any Image Restoration](../../CVPR2025/image_restoration/complexity_experts_are_task-discriminative_learners_for_any_image_restoration.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](../../ECCV2024/image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)

</div>

<!-- RELATED:END -->
