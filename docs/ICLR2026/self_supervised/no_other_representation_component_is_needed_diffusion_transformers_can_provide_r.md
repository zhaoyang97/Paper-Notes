# No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves

**会议**: ICLR2026  
**arXiv**: [2505.02831](https://arxiv.org/abs/2505.02831)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: diffusion transformer, self-representation alignment, DiT, SiT, image generation

## 一句话总结
提出 Self-Representation Alignment (SRA)，利用扩散 Transformer 内部从"差到好"的判别过程，将早层高噪声的表征对齐到晚层低噪声的表征，无需外部表征组件即可加速生成训练并提升质量。

## 研究背景与动机

1. **领域现状**：扩散 Transformer（DiT, SiT）是图像生成的主流架构。近期研究发现学习好的内部表征可以加速训练并提升质量。
2. **现有痛点**：现有方法要么需外部判别任务要么需大型预训练编码器（DINOv2、CLIP）来提供表征指导，限制了灵活性。
3. **核心矛盾**：表征指导对加速训练很重要，但对外部组件的依赖限制了使用场景。
4. **本文要解决什么？** 不依赖任何外部组件，仅利用扩散模型自身实现表征对齐。
5. **切入角度**：扩散模型去噪过程本质是"从差到好"——内部表征也遵循此趋势。
6. **核心idea一句话**：将学生网络早层高噪声条件下的表征对齐到教师（EMA）网络晚层低噪声条件下的表征。

## 方法详解

### 整体框架
在标准扩散生成训练上额外加自表征对齐损失。学生第 $m$ 层在时间步 $t$ 的输出经投影头后，与教师（EMA）第 $n$ 层（$n \geq m$）在时间步 $t-k$（更低噪声）的输出对齐。

### 关键设计

1. **"从差到好"的经验观察**: PCA 可视化和线性探测都表明随层数增加和噪声降低，特征质量渐进改善

2. **自表征对齐（SRA）**: $\mathcal{L}_{sa} = \mathbb{E}\left[\frac{1}{N}\sum_i\text{dist}(\mathbf{y}_*^{[i]}, j_\psi(\mathbf{y}^{[i]}))\right]$，利用噪声和层数两个维度的差到好梯度

3. **EMA 教师网络**: $\alpha=0.9999$ 不变，简单有效，不需要 SSL 中的稳定性技巧

### 损失函数
$\mathcal{L} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{sa}$

## 实验关键数据

### 主实验：ImageNet 256×256

| 方法 | 外部依赖 | FID 效果 |
|------|---------|---------|
| MaskDiT | MAE损失 | 较好 |
| REPA | DINOv2 | 好 |
| **SRA** | **无** | **可比REPA** |

### 消融实验

| 配置 | FID | 说明 |
|------|-----|------|
| 同层同时步对齐 | 差 | 无差异信号 |
| 早层→晚层同时步 | 中等 | 只利用层间梯度 |
| 早层高噪声→晚层低噪声 | 最佳 | 两个维度 |

### 关键发现
- 在 DiT 和 SiT 各尺寸上一致改善
- 性能可比拟 REPA（依赖 DINOv2），大幅优于 MaskDiT
- 表征对齐增强了判别能力

## 亮点与洞察
- "从差到好"洞察极为直觉——扩散模型去噪过程天然包含表征质量梯度
- 零外部依赖——不需要 DINOv2/CLIP，各种场景都可用
- EMA $\alpha=0.9999$ 不变的简单设计

## 局限性 / 可改进方向
- 层索引和时间偏移需要调参
- 主要在 ImageNet 256×256 验证

## 相关工作与启发
- **vs REPA**: 用外部 DINOv2 做指导，SRA 无外部依赖但效果可比
- **vs MaskDiT**: 用外部 MAE 判别任务，SRA 不需要额外目标
- **vs BYOL/DINO**: 自监督中的自蒸馏思想，SRA 适配到扩散模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "从差到好"洞察和自表征对齐设计都很新颖
- 实验充分度: ⭐⭐⭐⭐ 多尺寸、全面消融
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，分析深入
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型训练提供简洁有效加速方案
