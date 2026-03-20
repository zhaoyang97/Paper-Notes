# Infinite-Story: A Training-Free Consistent Text-to-Image Generation

**会议**: AAAI 2026  
**arXiv**: [2511.13002](https://arxiv.org/abs/2511.13002)  
**代码**: 无  
**领域**: 图像生成 / 一致性生成  
**关键词**: 一致性文本到图像生成, 视觉故事, 自回归生成, training-free, 风格一致性  

## 一句话总结
基于 scale-wise 自回归模型（Infinity），通过三个 training-free 技术——Identity Prompt Replacement（消除文本编码器的上下文偏差）、Adaptive Style Injection（参考图像特征注入）和 Synchronized Guidance Adaptation（同步 CFG 两个分支），实现了身份与风格一致的多图像生成，速度比扩散模型快 6 倍（1.72 秒/张）。

## 背景与动机
一致性文本到图像生成在视觉故事、漫画、角色驱动内容创作等场景中至关重要。现有方法存在两个问题：(1) 大多基于扩散模型，推理速度慢（通常>10秒/张），超出用户交互耐受阈值；(2) 现有工作主要关注身份一致性，忽视了风格一致性——同一角色在不同场景中的渲染风格、色调、背景风格可能完全不同（如 1Prompt1Story）。Scale-wise 自回归模型（Infinity 等）通过 next-scale 预测范式提供了更快的推理速度，但同样面临一致性挑战。

## 核心问题
如何在不需要额外训练的情况下，让 scale-wise 自回归 T2I 模型生成一组在身份（Identity）和风格（Style）上都保持一致的图像？挑战来自文本编码器的上下文偏差（不同 prompt 中相同身份描述产生不同语义理解），以及缺乏跨图像的视觉特征对齐机制。

## 方法详解

### 整体框架
基于 Infinity（2B 参数的 scale-wise 自回归模型，使用 Flan-T5 文本编码器），将 N 个 prompt 作为一个 batch 并行处理。第一个样本作为参考（anchor），其身份和风格特征传播到其余样本。三个技术分别作用于文本编码层和生成过程的早期自注意力层。

### 关键设计
1. **Identity Prompt Replacement (IPR)**：观察到文本编码器中"a dog"在不同上下文中（如"springing toward a frisbee"vs"on a porch swing"）会编码成不同的语义（柯基 vs 金毛）。IPR 将所有样本的身份 embedding 替换为参考样本的身份 embedding $T_{iden}^1$，同时对表情/场景 embedding 做归一化以保持比例关系：$\hat{T}_{exp}^n = \frac{\|T_{iden}^1\|}{\|T_{iden}^n\|} \cdot T_{exp}^n$。这在编码层消除了上下文偏差。

2. **Adaptive Style Injection (ASI)**：在早期生成步骤（S_early={2,3}）的自注意力层中，将所有样本的 Key 替换为参考样本的 Key，并基于余弦相似度自适应插值 Value：$\bar{V}_s^n = \alpha_s^n V_s^n + (1-\alpha_s^n) V_s^1$，其中 $\alpha_s^n = \lambda \cdot \text{sim}(V_s^1, V_s^n)$。相似度高的区域保留更多原始特征，相似度低的区域更多从参考中借鉴，实现自适应的外观和风格对齐。

3. **Synchronized Guidance Adaptation (SGA)**：ASI 只应用于 CFG 的条件分支会破坏条件/无条件分支的平衡，影响 prompt 忠实度。SGA 将相同的操作（使用条件分支计算的相同 α 权重）同步应用到无条件分支，恢复 CFG 平衡。

### 损失函数 / 训练策略
完全 training-free，无需训练或微调。所有参数冻结，仅在推理时修改注意力层的 K/V 特征。

## 实验关键数据
| 方法 | CLIP-I↑ | DreamSim↓ | CLIP-T↑ | DINO↑ | 时间(s/img) |
|------|---------|-----------|---------|-------|-------------|
| Infinite-Story | **0.8089** | **0.1834** | 0.8732 | **0.9267** | **1.72** |
| 1Prompt1Story | 0.7687 | 0.1993 | 0.8942 | 0.9117 | 22.57 |
| IP-Adapter | 0.7834 | 0.2266 | 0.8661 | 0.9243 | 10.40 |
| ConsiStory | 0.6895 | 0.2787 | 0.9019 | 0.8954 | 37.76 |
| Vanilla Infinity | 0.6965 | 0.2780 | 0.8836 | 0.8955 | 1.71 |

用户研究：58.4% 的参与者偏好 Infinite-Story（vs 18% 1Prompt1Story, 16.4% IP-Adapter, 7.2% OneActor）。

### 消融实验要点
- IPR 单独贡献：CLIP-I 从 0.6965 提升到 0.7119，DreamSim 从 0.2780 降到 0.2569
- 加入 ASI 后：DINO 大幅提升至 0.9242（风格一致性显著改善），CLIP-I 跃升至 0.8082
- 加入 SGA 后：CLIP-T 从 0.8625 提升到 0.8732（prompt 忠实度恢复），整体 S_H 最优
- λ 参数敏感性：λ=0.85 在一致性和 prompt 忠实度之间取得最佳平衡
- 在 Switti 和 HART 上也有效，证明方法可推广到其他 scale-wise 自回归模型

## 亮点
- **6× 推理加速**：1.72 秒/张 vs 扩散模型 10-38 秒/张，达到交互式应用的实用门槛
- **上下文偏差的发现和解决**：Identity Prompt Replacement 简洁优雅地解决了文本编码器中"相同描述因上下文不同产生不同语义"的问题
- **自适应插值权重**：ASI 通过余弦相似度自适应调整注入强度，避免了硬替换导致的细节丢失
- **完全 training-free**：三个技术都只在推理时操作注意力特征，零额外训练成本

## 局限性 / 可改进方向
- 依赖单一参考图像（anchor），如果 anchor 质量差会传播到整个 batch
- 身份一致性主要通过注意力层操作实现，对高度结构化或精细细节的控制有限
- 仅支持 scale-wise 自回归模型，未验证在扩散模型上的适用性
- CLIP-T 相比某些基线方法略低，说明一致性和 prompt 忠实度之间仍有 trade-off
- 未探索自适应 anchor 选择或修正机制

## 与相关工作的对比
- **vs 1Prompt1Story**：同为 training-free，但 1Prompt1Story 基于扩散模型（22.57s/img），且只关注身份一致性不关注风格一致性
- **vs ConsiStory/StoryDiffusion**：这些方法修改注意力权重实现身份一致，但推理极慢（24-38s），且风格一致性差
- **vs IP-Adapter**：IP-Adapter 需要参考图像且推理较慢，且 prompt 忠实度较低（过度受参考图影响）

## 启发与关联
- "在 CFG 的两个分支上同步操作"是一个通用的可控生成技巧，可用于其他 training-free 方法
- 上下文偏差问题不仅存在于 T2I，在 VLM 中也普遍存在（同一视觉概念在不同文本上下文中被不同理解）
- Scale-wise 自回归模型的生成速度优势值得关注，可能成为扩散模型的有力替代

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个在 scale-wise 自回归模型上实现一致性 T2I 的 training-free 方法
- 实验充分度: ⭐⭐⭐⭐ 多指标评估、用户研究、消融完整，跨模型泛化验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，上下文偏差的可视化分析直观有说服力
- 价值: ⭐⭐⭐⭐ 推理速度的大幅提升使一致性 T2I 达到实用水平，对视觉故事应用有直接价值
