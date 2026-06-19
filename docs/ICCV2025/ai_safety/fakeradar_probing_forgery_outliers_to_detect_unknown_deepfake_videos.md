---
title: >-
  [论文解读] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos
description: >-
  [ICCV 2025][AI安全][Deepfake检测] 提出FakeRadar深度伪造视频检测框架，通过Forgery Outlier Probing在特征空间中主动生成模拟未知伪造的异常值样本，并设计Outlier-Guided Tri-Training三分类优化策略，在跨数据集/跨操纵类型评估中显著超越现有方法。
tags:
  - "ICCV 2025"
  - "AI安全"
  - "Deepfake检测"
  - "跨域泛化"
  - "异常值探测"
  - "对比学习"
  - "CLIP"
---

# FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos

**会议**: ICCV 2025  
**arXiv**: [2512.14601](https://arxiv.org/abs/2512.14601)  
**代码**: 无  
**领域**: AI安全  
**关键词**: Deepfake检测, 跨域泛化, 异常值探测, 对比学习, CLIP

## 一句话总结
提出FakeRadar深度伪造视频检测框架，通过Forgery Outlier Probing在特征空间中主动生成模拟未知伪造的异常值样本，并设计Outlier-Guided Tri-Training三分类优化策略，在跨数据集/跨操纵类型评估中显著超越现有方法。

## 研究背景与动机
- **领域现状**：深度伪造技术快速发展，产生高度逼真的面部伪造视频，大量检测方法被提出，包括图像级和视频级方法
- **现有痛点**：现有方法依赖特定伪造痕迹（边界不一致、眨眼异常、纹理不规则等），仅对已知伪造类型有效，面对新型生成技术（如扩散模型）时泛化能力严重退化
- **核心矛盾**：真实数据呈紧凑密集分布，而伪造数据因操纵方式多样形成稀疏分散的聚簇——在已知伪造模式上训练的分类器无法覆盖未知伪造模式空间
- **切入角度**：从"被动学习已知伪造模式"转向"主动探测未知伪造区域"，类比雷达系统通过频谱探测扫描未知目标
- **核心idea**：利用CLIP预训练模型的深层特征先验，通过动态子簇建模和聚簇条件异常值生成来"预探索"特征空间中未知伪造可能出现的区域

## 方法详解

### 整体框架
FakeRadar基于CLIP ViT-B/16冻结骨干网络，插入ST-Adapter进行参数高效微调。包含两大核心组件：(1) Forgery Outlier Probing负责在特征空间生成模拟未知伪造的异常值样本；(2) Outlier-Guided Tri-Training通过三分类（Real/Fake/Outlier）联合优化模型，推理时将Fake和Outlier合并为"伪造"类。

### 关键设计
1. **Forgery Outlier Probing (FOP)**:

    - 功能：对训练样本的特征空间分布进行精细建模，在子簇边界处生成异常值样本模拟未知伪造
    - 核心思路：
        - **动态子簇建模**：将真实类和4种伪造类分别视为独立类别，用主聚簇网络学习GMM分布的软分配，通过KL散度损失 $\mathcal{L}_{main} = \sum_i KL(\mathbf{r}_i \| \mathbf{r}_i^E)$ 对齐聚簇分配与GMM责任。子聚簇网络尝试将每个簇进一步分裂为两个子簇，使用Hastings比 $H_s$ 决定分裂/合并
        - **聚簇条件异常值生成**：从子簇分布的ε-似然区域采样，生成位于簇边界附近的异常值 $\mathcal{V}_k$，条件为其在子簇高斯分布下的概率密度小于阈值ε
    - 设计动机：固定K=5不足以捕获伪造类的内部多模态结构，动态分裂/合并能更精确地建模分布；在边界处生成异常值能模拟"新型伪造偏移"

2. **Outlier-Guided Tri-Training (OGTT)**:

    - 功能：联合优化骨干网络和三分类器，显式区分Real、Fake和Outlier三类
    - 核心思路：
        - **Outlier-Driven Contrastive Loss**：基于InfoNCE的对比损失 $\mathcal{L}_{con}$，最大化样本与所属子簇中心的相似度，同时最小化与其他子簇中心及异常值的相似度
        - **Outlier-Conditioned Cross-Entropy Loss**：三分类交叉熵 $\mathcal{L}_{cls} = -\sum_{c} y_c \log p_c$，确保模型对三类（特别是Outlier不误判为Real）有清晰决策边界
        - 总损失：$\mathcal{L}_{total} = \mathcal{L}_{con} + \lambda \mathcal{L}_{cls}$，$\lambda=0.5$
    - 设计动机：三分类让模型在训练时独立标注未知伪造，推理时Fake+Outlier合并为伪造类；对比损失增强类间分离，交叉熵确保决策边界清晰

3. **模型适配与推理**:

    - 功能：在冻结CLIP骨干上插入ST-Adapter进行参数高效微调
    - 核心思路：$\text{ST-Adapter}(x) = x + \text{ReLU}(\text{Conv3D}(xW_{down}))W_{up}$，3D卷积捕获时空特征，仅引入少量额外参数
    - 设计动机：保留CLIP预训练的丰富语义特征，同时适配深度伪造检测的时空模式

### 损失函数 / 训练策略
- 总损失：对比损失 + 0.5 × 三分类交叉熵
- 每个mini-batch包含16个训练样本和16个异常值样本
- Adam优化器，余弦学习率调度，初始学习率1e-4，60个epoch
- 每个视频采样4个时间片段，每个包含12个连续帧，面部图像统一为224×224

## 实验关键数据

### 主实验 (跨数据集评估, AUC%)

| 方法 | 骨干 | FF++ | CDFv2 | DFDCP | DFDC | DFD |
|------|------|------|-------|-------|------|-----|
| LSDA | EfficientNet | - | 91.1 | 81.2 | 77.0 | 95.6 |
| TALL | Swin Trans. | 99.9 | 90.8 | - | 76.8 | - |
| AltFreezing | 3D ResNet | 99.7 | 89.5 | - | - | 93.7 |
| **FakeRadar** | **ViT-B/16** | **99.1** | **91.7** | **88.5** | **84.1** | **96.2** |

### 消融实验

| 配置 | CDFv2 | DFDC | DFDCP | DFD | 平均 |
|------|-------|------|-------|-----|------|
| FakeRadar完整 | 91.6 | 84.1 | 88.5 | 96.2 | 90.1 |
| 移除ODCL | 89.9 | 81.2 | 85.6 | 94.9 | 87.9 |
| 移除OCCE | 88.6 | 80.9 | 87.7 | 94.7 | 88.0 |
| 移除FOP | 88.8 | 80.2 | 86.7 | 94.5 | 87.6 |
| 仅二分类 | 88.4 | 78.4 | 85.1 | 94.3 | 86.7 |
| 无微调 | 88.2 | 78.3 | 84.8 | 94.2 | 86.4 |

**模型变体** (跨数据集AUC%):

| 变体 | FF++ | CDFv2 | DFDCP | DFDC | DFD |
|------|------|-------|-------|------|-----|
| Frozen (无微调) | 55.2 | 60.0 | 59.0 | 55.2 | 57.4 |
| Supervised (二分类) | 98.2 | 88.2 | 84.8 | 78.3 | 94.2 |
| Proposed (完整) | 99.1 | 91.7 | 88.5 | 84.1 | 96.2 |

### 关键发现
- FakeRadar在DFDC上超越UCF和LTTD分别3.6%和3.7% AUC，跨域泛化提升显著
- 跨操纵类型评估中(用1种训练、测试其余3种)，FakeRadar平均AUC优于所有方法，F2F训练场景下超越DCL 7.41%
- 动态子簇建模在训练初期(0-8 epoch)子簇数量剧烈波动，约10 epoch后稳定在3个
- 三分类器对DFDC（与训练集伪造类型不同）的纠错率在第10 epoch达约40%，显著高于FF++测试集的约5%
- 固定K=5 vs 动态K：完整模型90.1% vs 固定K 87.4%，差距2.7%

## 亮点与洞察
- "主动探测"的范式转变有启发性：不是被动学习已知模式，而是主动在特征空间中探索未知区域
- 异常值生成在特征空间而非像素空间进行，避免了合成伪造图像的困难
- 三分类(Real/Fake/Outlier)训练 + 二分类(Real/Fake)推理的设计简洁且有效
- 动态子簇的分裂/合并过程在训练中自适应调整，无需手动设定最终簇数
- t-SNE可视化清晰展示了FakeRadar学到的更紧凑、边界更清晰的特征分布

## 局限与展望
- 仅在FaceForensics++(HQ)上训练，未探讨使用更多/更新伪造类型训练数据的效果
- 异常值采样的ε阈值为超参数，对不同数据集的敏感性未充分讨论
- 推理时将Outlier简单合并到Fake类，未利用Outlier的置信度信息进行更精细判断
- 仅使用ViT-B/16骨干，更大模型(ViT-L)是否能进一步提升泛化能力待探索
- 缺少对抗鲁棒性评估（如面对蓄意规避的对抗样本）

## 相关工作与启发
- VOS (Virtual Outlier Synthesis) 的思想被有效迁移到深度伪造检测领域
- ST-Adapter的参数高效微调策略成功保留CLIP的通用特征同时适配下游任务
- 子簇分裂/合并的动态建模思路（受Dirichlet过程启发）值得在其他分布建模场景借鉴
- 与SBI等数据合成方法正交——本文在特征空间而非像素空间合成，两者可结合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从被动检测到主动探测的范式转变+特征空间异常值生成+三分类训练策略，整体思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 跨数据集、跨操纵、消融、模型变体、特征可视化、子簇演化分析全面深入
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，"雷达探测"类比直观，但部分公式符号可进一步精简
- 价值: ⭐⭐⭐⭐⭐ 实际意义重大——针对不断涌现的新型伪造技术，提供了一种不依赖特定伪造模式的泛化检测方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[CVPR 2026\] SAGA: Source Attribution of Generative AI Videos](../../CVPR2026/ai_safety/saga_source_attribution_of_generative_ai_videos.md)
- [\[CVPR 2026\] Beyond \[CLS\] Token: Query-Driven Token-Level Forgery Purification for Generalizable Deepfake Detection](../../CVPR2026/ai_safety/beyond_cls_token_query-driven_token-level_forgery_purification_for_generalizable.md)
- [\[CVPR 2026\] Detect Any AI-Counterfeited Text Image](../../CVPR2026/ai_safety/detect_any_ai-counterfeited_text_image.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)

</div>

<!-- RELATED:END -->
