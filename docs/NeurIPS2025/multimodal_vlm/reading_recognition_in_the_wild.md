---
title: >-
  [论文解读] Reading Recognition in the Wild
description: >-
  [NeurIPS 2025][多模态VLM][阅读识别] 提出了阅读识别新任务及首个大规模多模态"野外阅读"数据集（100小时），利用RGB、眼动和IMU三种互补模态的轻量级Transformer模型，在智能眼镜上实现实时阅读检测。 智能眼镜作为未来AI个人助手的载体，需要理解用户的物理上下文。阅读是人类最重要的信息获取方式…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "阅读识别"
  - "可穿戴设备"
  - "多模态融合"
  - "眼动追踪"
  - "第一人称视觉"
---

# Reading Recognition in the Wild

**会议**: NeurIPS 2025  
**arXiv**: [2505.24848](https://arxiv.org/abs/2505.24848)  
**代码**: [Project Aria](https://www.projectaria.com/datasets/reading-in-the-wild/)  
**领域**: 多模态VLM  
**关键词**: 阅读识别, 可穿戴设备, 多模态融合, 眼动追踪, 第一人称视觉

## 一句话总结

提出了阅读识别新任务及首个大规模多模态"野外阅读"数据集（100小时），利用RGB、眼动和IMU三种互补模态的轻量级Transformer模型，在智能眼镜上实现实时阅读检测。

## 研究背景与动机

智能眼镜作为未来AI个人助手的载体，需要理解用户的物理上下文。阅读是人类最重要的信息获取方式之一，使AI具备"知道用户何时在阅读"的能力，对于构建上下文感知的个人助手至关重要。

然而，阅读识别面临两大核心挑战：

**问题的病态性**：视野中有文字不等于用户在阅读（如路过广告牌），仅靠视觉信息存在歧义；

**效率约束**：可穿戴设备有功耗、带宽和散热等硬件限制，不可能全程运行OCR或VLM等重型模型。

现有工作的不足：
- 眼动追踪方法（如Kelton等）依赖手工特征（如注视点、扫视检测），且仅在受控环境中实验；
- 第一人称视频数据集（Ego4D、Ego-Exo4D）中阅读样本极少且缺乏多样性；
- 认知研究数据集（ZuCo、InteRead）仅有屏幕前的受控场景，缺少RGB信息。

阅读识别可作为轻量级代理信号，仅在检测到阅读时才触发重型OCR/VLM模型，大幅节省计算资源。

## 方法详解

### 整体框架

给定时刻 $t$，模型预测用户是否在阅读的置信度 $s_t \in [0,1]$，输入包含三种模态：眼动轨迹 $g$、RGB图像裁剪 $I_t$ 和头部姿态（IMU）$z$。框架为一个灵活的多模态Transformer，可接受任意模态组合作为输入。

### 关键设计

1. **眼动（Gaze）编码**：使用3D眼动点的时间差分作为输入表示（而非2D投影或视网膜图像），通过3层1D卷积（kernel=9, dim=32）编码为特征token。差分处理使模型聚焦于眼动变化模式而非绝对位置，增强泛化性。

2. **RGB裁剪策略**：基于人眼中央凹仅覆盖约2°视角的事实，仅裁剪注视点周围 5° FoV（64×64像素，仅占完整图像的1/484），通过3层2D卷积编码。这一设计既提供了足够的视觉上下文，又极大减少了计算量和隐私泄露风险。

3. **头部姿态（IMU/VIO）**：利用视觉-惯性里程计（VIO）的6DoF输出判断头部运动模式。虽然单独性能有限，但可消除歧义（如区分阅读和水平转头）。

4. **模态Dropout训练**：训练时随机丢弃整个模态，使得：(i) 较少使用的模态也能得到充分训练；(ii) 推理时即使缺少某些模态也能正常工作。

5. **跨语言泛化**：针对不同阅读方向语言（如中文↓、阿拉伯文←），在推理时对眼动数据进行90°旋转或水平翻转，无需重新训练即可适配。

### 损失函数 / 训练策略

- 二分类交叉熵损失，预测阅读/非阅读
- Adam优化器，学习率 $1 \times 10^{-3}$，训练10个epoch
- 模态Dropout使一、二、三模态的使用概率相等
- 训练时加入小比例旋转增强以适应垂向文本
- 总参数量仅137K，单GPU即可训练

## 实验关键数据

### 主实验

| 模态组合 | 准确率(%) | F1(%) | $P_{R=0.9}$(%) |
|---------|----------|-------|---------------|
| 仅Gaze | 82.3 | 84.5 | 79.8 |
| 仅RGB | 82.2 | 83.7 | 76.5 |
| 仅IMU | 74.7 | 80.0 | 71.9 |
| Gaze+RGB | 84.9 | 86.5 | 83.6 |
| Gaze+IMU | 83.5 | 85.2 | 82.3 |
| RGB+IMU | 86.0 | 87.8 | 87.3 |
| **三模态** | **86.9** | **88.1** | **88.0** |

三模态比最佳单模态提升+4.6%准确率，验证了模态互补性。

### 消融实验

| 配置 | 准确率(%) | F1(%) | 说明 |
|------|----------|-------|------|
| 3D point (d/dt) | 82.3 | 84.5 | ✓ 最优Gaze表示 |
| 2D projection | 79.8 | 81.3 | 2D投影损失3D信息 |
| 60Hz采样 | 82.3 | 84.5 | 高频最优 |
| 10Hz采样 | 80.4 | 82.9 | 降频仍可用 |
| FoV 5° (64px) | 82.2 | 83.7 | 最优效率-精度平衡 |
| XS模型 (6K参数) | 82.0 | 83.6 | 极小模型仍有效 |
| XL模型 (1M参数) | 88.5 | 90.1 | 强模型效果更好 |

### 泛化实验

| 场景 | 准确率(%) |
|------|----------|
| Columbus零样本（三模态） | 82.9 |
| 孟加拉语（左→右） | 93.0 |
| 中文（↓）+ 旋转增强 | 85.1 (+49.6) |
| 阿拉伯语（←）+ 翻转增强 | 51.5 (+30.5) |
| Seattle→EGTEA泛化 | 87.7 |
| EGTEA→Seattle泛化 | 62.9 |

### 关键发现

- Gaze和RGB的失败案例互补：Gaze擅长低光照/远距离场景，RGB擅长短文本阅读检测
- IMU作为辅助传感器单调提升其他模态性能（+1.3%~+2.6%）
- 难负样本（文字存在但未被阅读）准确率仅74.7%，是主要挑战
- 实时检测延迟：Gaze+RGB+IMU模型约0.72秒
- 仅66K参数的S模型已有86.3%准确率

## 亮点与洞察

1. **任务定义新颖**：将"阅读识别"从受控实验室推向野外真实场景，是第一个如此大规模的阅读识别基准
2. **极致的效率设计**：仅裁剪图像0.2%区域、137K参数模型、可在Aria Gen 2眼镜上运行4小时以上
3. **可扩展的数据收集协议**：通过语音标注（"开始阅读！"/"结束阅读！"）+ WhisperX自动获取时间戳，避免人工标注
4. **隐私保护导向**：使用Gaze可完全避免采集完整RGB图像，减少对旁观者的视觉入侵

## 局限与展望

- 非典型阅读场景（边写边读55.5%、非文本阅读65.8%）表现较差
- 短文本（如路标）难以在时间窗口内产生明显眼动模式
- 阿拉伯语等右→左阅读方向的泛化仍有提升空间（翻转后仅51.5%）
- 2秒时间窗口可能不足以区分精读/略读/扫读等精细阅读模式

## 相关工作与启发

- 连接了计算机视觉（第一人称活动识别）和认知科学（阅读理解研究）两个领域
- 为未来多模态感知的"按需激活"范式提供了实践案例：轻量模型做门控、重型模型按需执行
- 数据集发布将推动阅读辅助工具（针对失读症儿童和低视力人群）的研究

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义野外阅读识别任务，提出全新数据集和基准
- 实验充分度: ⭐⭐⭐⭐⭐ 大量消融实验、跨语言/跨数据集泛化、实时部署验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务动机阐释充分
- 实用价值: ⭐⭐⭐⭐⭐ 直接适用于智能眼镜产品，Meta内部项目背景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trust but Verify: Programmatic VLM Evaluation in the Wild](../../ICCV2025/multimodal_vlm/trust_but_verify_programmatic_vlm_evaluation_in_the_wild.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](../../CVPR2026/multimodal_vlm/joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[ECCV 2024\] Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](../../ECCV2024/multimodal_vlm/nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)
- [\[CVPR 2025\] Recognition-Synergistic Scene Text Editing](../../CVPR2025/multimodal_vlm/recognition-synergistic_scene_text_editing.md)

</div>

<!-- RELATED:END -->
