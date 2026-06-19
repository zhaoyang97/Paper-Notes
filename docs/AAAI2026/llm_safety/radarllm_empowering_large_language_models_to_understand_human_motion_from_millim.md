---
title: >-
  [论文解读] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence
description: >-
  [AAAI 2026][LLM安全][毫米波雷达] 提出 RadarLLM，首个利用大语言模型从毫米波雷达点云进行语义级人体运动理解的端到端框架，包含基于 Aggregate VQ-VAE 的运动引导雷达分词器和雷达感知语言模型，并通过物理感知仿真管线生成大规模雷达-文本配对数据。 人体运动理解在老年护理、智能家居、健康监测…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "毫米波雷达"
  - "大语言模型"
  - "人体运动理解"
  - "向量量化"
  - "隐私保护"
---

# RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence

**会议**: AAAI 2026  
**arXiv**: [2504.09862](https://arxiv.org/abs/2504.09862)  
**代码**: [inowlzy.github.io/RadarLLM](https://inowlzy.github.io/RadarLLM/)  
**领域**: 3D视觉  
**关键词**: 毫米波雷达, 大语言模型, 人体运动理解, 向量量化, 隐私保护

## 一句话总结

提出 RadarLLM，首个利用大语言模型从毫米波雷达点云进行语义级人体运动理解的端到端框架，包含基于 Aggregate VQ-VAE 的运动引导雷达分词器和雷达感知语言模型，并通过物理感知仿真管线生成大规模雷达-文本配对数据。

## 研究背景与动机

人体运动理解在老年护理、智能家居、健康监测等场景至关重要。传统基于视觉的系统受**光照变化、遮挡和隐私问题**限制，不适合长期实际部署。

**毫米波（mmWave）雷达**作为隐私保护且环境鲁棒的替代方案，能在低光照、遮挡、雨雾等恶劣条件下工作，且不捕获视觉身份信息。然而，现有雷达方法主要聚焦于**分类或回归任务**（如活动识别、姿态估计），局限于预定义标签集，缺乏生成细粒度运动描述的能力。

将 LLM 应用于雷达数据面临**两个核心挑战**：

**稀疏噪声点云的时空建模**：雷达点云远比 LiDAR 或 RGB-D 更稀疏（每帧仅128点），信噪比低，难以提取有意义的语义特征。

**雷达与语言的语义鸿沟**：雷达信号的底层物理特性（多普勒频移、距离-速度信息）与自然语言的高级语义之间存在巨大差距。

此外，**配对雷达-文本数据极度稀缺**：现有雷达数据集规模小（<9小时、<40人、<27类），且完全缺少自然语言标注。

作者提出通过三个创新解决这些挑战：运动引导雷达分词器、雷达感知语言模型、物理感知数据合成管线。

## 方法详解

### 整体框架

RadarLLM 包含三个核心组件（Figure 3）：
1. **雷达-文本数据集准备**：物理感知仿真管线生成配对数据
2. **运动引导雷达分词器**：将稀疏雷达点云序列编码为离散语义token
3. **雷达感知语言模型**：跨模态对齐 + 文本生成

### 关键设计

#### 1. **物理感知虚拟数据合成（Physics-Aware Data Synthesis）**

**功能**：从运动-文本数据集（HumanML3D, 13308条 SMPL-X 序列含文本标注）合成逼真的雷达点云-文本配对。

合成管线（Figure 2）：
- **IF信号仿真**：在渲染的人体网格和虚拟雷达天线之间进行射线追踪。采用 RF 自适应采样聚焦人体区域，使用物理光学积分（POI）累积射线信息。加入高斯噪声模拟真实噪底：

$$R'_{IF}(t) = R_{IF}(t) + \sqrt{P_{\text{signal}} / 10^{\text{SNR}/10}} \cdot \epsilon(t)$$

- **点云生成**：Range-FFT → Doppler-FFT → 静态杂波去除 → 选择强度最高的128个点（保证每帧点数一致）

最终每个点的6D特征向量：$\mathbf{p}_m = [x, y, z, r, v, 10\log_{10}(|D_m|)]^T$，包含3D坐标、径向距离、速度和对数多普勒强度。

**设计动机**：缺乏大规模配对数据是训练 LLM 的根本瓶颈，物理感知仿真避免了人工标注的高成本。

#### 2. **运动引导雷达分词器（Motion-Guided Radar Tokenizer）**

**功能**：将稀疏、噪声的雷达点云序列压缩为 LLM 可处理的离散语义token。基于创新的 Aggregate VQ-VAE 架构（Figure 4）。

**三个阶段**：

**(1) 模板先验分组（Template-Prior Grouping）**：
- 在人体边界框模板的 $N_x \times N_y \times N_z$ 确定性网格上初始化 $N_g$ 个锚点
- 对每个锚点周围的邻域点进行时序聚合
- 用 P4Conv 编码器 $\mathbf{E}$ 提取分组特征 $\mathbf{F}_{group} \in \mathbb{R}^{L \times N_g \times C}$

**设计动机**：解决帧间点位置和数量不一致的问题，通过确定性模板建立稳定的时序身体区域关联。

**(2) 掩码上下文聚合（Masked Context Aggregation）**：
- 随机掩码 50% 的锚点轨迹，得到可见特征 $\mathbf{F}_{vis}$
- Transformer 解码器通过交叉注意力重建被掩码特征：$\mathbf{F}_{msk} = D(\mathbf{F}_{vis})$
- 合并得到 $\mathbf{F}_{all} = [\mathbf{F}_{vis}, \mathbf{F}_{msk}]$
- 通过嵌入损失将雷达特征与配对的运动语义特征 $\mathbf{F}_{mot}$ 对齐

**设计动机**：掩码策略迫使模型学习身体各部位之间的依赖关系，运动语义引导加速特征学习。

**(3) 聚合量化（Aggregated Quantization）**：
- 将每个时间步的 $\mathbf{F}^t_{all}$ 映射到可训练码本 $\mathcal{Z} = \{\mathbf{z}_k\}_{k=1}^K \subset \mathbb{R}^{512 \times 512}$ 的最近码字：

$$\mathbf{z}_t = \arg\min_{\mathbf{z}_k \in \mathcal{Z}} \|\mathbf{F}^t_{all} - \mathbf{z}_k\|_2$$

分词器总损失：

$$\mathcal{L}_{VQ} = \mathcal{L}_{rec} + \mathcal{L}_{emb} + \mathcal{L}_{commit}$$

- $\mathcal{L}_{rec}$：Chamfer Distance 重建损失（重建被掩码的点云管）
- $\mathcal{L}_{emb} = \|\mathbf{F}_{all} - \mathbf{F}_{mot}\|_2^2$：运动引导嵌入损失
- $\mathcal{L}_{commit}$：码本承诺损失（含 stop-gradient）

#### 3. **雷达感知语言模型（Radar-Aware Language Model）**

**功能**：将雷达token与文本token在统一空间中对齐，自回归生成运动描述。

基于 T5 架构，统一词表 $\mathcal{V} = \mathcal{V}_{\text{text}} \cup \mathcal{V}_{\text{radar}}$（32768 WordPieces + $K$ 个雷达token + 特殊标记）。

**两阶段训练**：

**(1) 预训练阶段**——三任务联合训练：
- **雷达预测**：掩码15%雷达token → 预测原始token（$\mathcal{L}_{\text{pred}}$）
- **雷达→文本**：编码雷达token → 解码生成文本（$\mathcal{L}_{\text{r2t}}$）
- **文本→雷达**：编码文本 → 自回归生成雷达token（$\mathcal{L}_{\text{t2r}}$）

$$\mathcal{L}_{\text{pretrain}} = \lambda_1 \mathcal{L}_{\text{pred}} + \lambda_2 \mathcal{L}_{\text{r2t}} + \lambda_3 \mathcal{L}_{\text{t2r}}$$

**(2) 指令微调阶段**：使用指令感知提示（如"Describe the motion <Motion_Placeholder>…"），通过相似度损失精细对齐。

### 损失函数 / 训练策略

- 分词器：100 epochs，lr = 3.5×10⁻⁴
- 语言模型预训练：300 epochs，lr = 2×10⁻⁴
- 指令微调：100 epochs
- 统一 batch size = 16，RTX3090 单卡训练

## 实验关键数据

### 主实验

#### 雷达到文本生成性能

| 模型 | 数据域 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE |
|------|--------|---------|--------|--------|--------|-------|-----------|--------|
| AvatarGPT* | Virtual | 30.0 | 36.3 | 5.0 | 28.3 | 6.8 | 82.4 | 88.7 |
| Video-LLaMA2* | Virtual | 26.7 | 35.2 | 3.6 | 30.4 | 4.2 | 81.0 | 88.4 |
| MotionGPT* | Virtual | 29.4 | 37.6 | 5.0 | 26.1 | 6.5 | 82.6 | 88.9 |
| **RadarLLM** | **Virtual** | **36.0** | **48.0** | **11.4** | **33.7** | **8.3** | **83.3** | **89.6** |
| AvatarGPT* | Real | 28.8 | 38.1 | 4.2 | 25.6 | 5.6 | 81.4 | 88.1 |
| **RadarLLM** | **Real** | **28.8** | **44.2** | **5.0** | **25.7** | **4.0** | **81.4** | **88.1** |

虚拟数据上 RadarLLM 全面领先：ROUGE-L +20.0%，BLEU-4 +128%，CIDEr +22.1%。真实数据上保持竞争力。

### 消融实验

| 配置 | ROUGE-1 | ROUGE-L | BLEU-4 | CIDEr | 说明 |
|------|---------|---------|--------|-------|------|
| w/o 模板锚点 | 27.9 | 25.7 | 3.8 | 3.2 | -27.3% ROUGE-1 |
| w/o 掩码训练 | 35.0 | 32.4 | 8.7 | 11.3 | -23.7% BLEU-4 |
| w/o 嵌入损失 | 28.6 | 26.5 | 4.2 | 3.8 | -54.2% CIDEr |
| **RadarLLM完整** | **38.4** | **36.0** | **11.4** | **8.3** | 最优 |

| LLM 架构 | 参数量 | FPS↑ | ROUGE-L↑ | SimCSE↑ | 说明 |
|---------|-------|------|---------|---------|------|
| T5-small | 60M | **97.0** | 36.0 | 89.6 | 最平衡选择 |
| GPT2-M | 355M | 72.7 | 35.4 | 89.5 | 速度与质量折中 |
| Deepseek-R1 | 1.8B | 53.6 | **37.4** | **89.9** | 最高质量但慢 |

| 训练策略 | ROUGE-L | BLEU-1 | METEOR | BERTScore |
|---------|---------|--------|--------|-----------|
| 仅 R→T | 33.0 | 42.8 | 31.2 | 82.5 |
| R→T & T→R | 33.0 | 43.1 | 31.2 | 82.5 |
| R→T & R-Pred | 33.9 | 43.2 | 32.4 | 82.9 |
| **全部任务** | **36.0** | **48.0** | **33.7** | **83.3** |

三任务联合训练相比仅R→T提升 ROUGE-L +9.1%、BLEU-1 +12.2%、METEOR +8.0%。

### 关键发现

1. **端到端优于两阶段**：RadarLLM 端到端方案优于先做 HPE 再接视觉/运动LLM的两阶段方案。
2. **模板先验至关重要**：去除模板锚点后 ROUGE-1 下降 27.3%，说明稀疏雷达点云需要结构先验。
3. **多任务训练显著有效**：双向翻译+掩码预测的联合训练全面提升性能。
4. **恶劣环境下保持语义能力**：在雨、烟、低光照、遮挡条件下 ROUGE-L 仅下降 14.2%（28.8→24.7），SimCSE 仅下降 1.4%。

## 亮点与洞察

1. **开辟新范式**：从"预定义标签分类"到"自然语言描述"，RadarLLM 本质上改变了雷达运动理解的范式。
2. **物理感知数据合成**：完绝解决了配对数据缺乏的瓶颈，合成13K+运动序列的雷达-文本数据。
3. **Aggregate VQ-VAE 设计精巧**：模板分组→掩码聚合→运动引导量化的层层推进设计，每一步都有明确的技术动机。
4. **隐私保护 + 环境鲁棒**：毫米波雷达天然满足这两个日益重要的需求。

## 局限与展望

1. 仿真参数针对特定雷达硬件（TI AWR1843BOOST）校准，跨硬件泛化有待验证。
2. 合成数据不包含环境上下文和人-物交互，限制了场景理解的丰富度。
3. 真实数据集规模有限（375序列），大规模真实评估仍缺乏。
4. 当前仅支持单人运动描述，多人场景可作为重要扩展方向。
5. T5-small 在资源受限下表现好但语义精度略低，更大模型的全微调有待探索。

## 相关工作与启发

- **与 MotionGPT、AvatarGPT 的区别**：这些方法从SMPL-X骨架或视频生成文本，而 RadarLLM 直接从原始雷达点云出发，避免了中间HPE步骤的误差累积。
- **与 PointLLM 的类比**：PointLLM 将3D对象点云接入LLM，RadarLLM 同理但处理的是时序稀疏雷达点云，挑战更大。
- 物理感知仿真的数据增强思路可推广到其他传感器模态（超声、WiFi等）。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首创性工作，雷达+LLM方向的开拓者，VQ-VAE和数据管线设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 虚拟/真实双评估、消融完整、恶劣环境测试亮眼
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，各模块动机明确，补充材料详尽
- **实用价值**: ⭐⭐⭐⭐ — 隐私保护和恶劣环境下的语义运动理解有实际需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)
- [\[CVPR 2025\] LLM4SVG: Empowering LLMs to Understand and Generate Complex Vector Graphics](../../CVPR2025/llm_safety/empowering_llms_to_understand_and_generate_complex_vector_graphics.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)

</div>

<!-- RELATED:END -->
