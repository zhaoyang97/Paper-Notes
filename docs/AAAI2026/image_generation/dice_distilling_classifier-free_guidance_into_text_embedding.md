# DICE: Distilling Classifier-Free Guidance into Text Embeddings

**会议**: AAAI 2026 **(Oral)**
**arXiv**: [2502.03726](https://arxiv.org/abs/2502.03726)
**代码**: [https://github.com/zju-pi/dice](https://github.com/zju-pi/dice)
**领域**: 图像生成 / 扩散模型加速
**关键词**: Classifier-Free Guidance, 文本嵌入蒸馏, 扩散模型加速, CFG-free 采样, text embedding sharpening

## 一句话总结

提出 DICE，训练一个仅 2M 参数的轻量 sharpener 将 CFG 的引导效果蒸馏进 text embedding，使无引导采样达到与 CFG 同等的生成质量、推理计算量减半，在 SD1.5 多个变体、SDXL 和 PixArt-α 上全面验证有效，是 AAAI 2026 口头报告论文。

## 研究背景与动机

文本到图像扩散模型依赖 Classifier-Free Guidance (CFG) 来提升图文对齐和图像质量。CFG 在每个采样步需要同时执行 conditional 和 unconditional 两次前向传播，计算量翻倍。对于 SDXL（2.6B 参数）等大模型，这严重制约实时应用。现有 CFG 蒸馏方案均有明显不足：

| 现有方法 | 可训练参数 | 核心缺陷 |
|---------|-----------|---------|
| Guided Distillation (GD) | 859M（全模型微调） | 微调后的模型无法迁移到新场景变体 |
| Plug-and-Play (PnP) | 361M（辅助模型） | 推理时需多次操作，实际加速比打折 |
| Scaling embedding | 0 | 最优缩放因子因模型和 prompt 而异，且质量无法达到 CFG 水平 |

DICE 的核心 insight：先通过缩放实验验证 text embedding 空间存在可增强图文对齐的修正方向，再进一步发现 **CFG 的本质作用是锐化 text embedding 的特定分量**（主要是语义无关的 padding 部分），因此可训练一个极轻量 sharpener 直接在嵌入空间完成锐化，完全跳过 CFG 的双重前向传播。

## 方法详解

### 整体框架

DICE 在文本编码器之后插入一个轻量 sharpener $r_\phi$，将原始 text embedding $\mathbf{c}$ 修正为锐化版本 $\mathbf{c}_\phi = \mathbf{c} + \alpha \cdot r_\phi(\mathbf{c}, \mathbf{c}_\text{null})$。Sharpener 仅含 2M 参数，独立于扩散模型运行、仅需一次嵌入修正。训练完成后，采样时只需用 $\mathbf{c}_\phi$ 做单次去噪前向传播（无 CFG），即可达到 CFG 等效的生成质量。

### 关键设计

1. **Scaling 验证实验**：将 text embedding 乘以缩放因子 $s$，在 DreamShaper 上 $s=1.3$ 时无引导生成质量显著提升。但最优 $s$ 因模型/prompt 而异，且简单缩放无法达到 CFG 水平 → 需要学习动态的细粒度修正
2. **CFG 方向蒸馏**：以 CFG 增强后的噪声预测 $\epsilon_\theta^{\omega,\mathbf{c}_\text{null}}(\mathbf{x}_t, \mathbf{c})$ 为 teacher，使用锐化嵌入的单次推理 $\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_\phi)$ 为 student，训练目标是使两者噪声预测方向一致
3. **Sharpening 机制分析**：text embedding 由语义 token 和 padding token 组成。DICE 发现**锐化主要放大语义无关的 padding 分量**，保留核心语义信息的同时增强细粒度细节。仅用锐化后的 padding embedding 即可大幅提升生成质量
4. **Negative Prompt 支持**：通过 $\mathbf{c}_\phi = \mathbf{c} + \alpha r_\phi(\mathbf{c}, \mathbf{c}_n) - \beta(\mathbf{c}_n - \mathbf{c}_\text{null})$ 集成负提示嵌入，$\beta$ 控制语义偏移强度。训练时随机采样负提示，增强鲁棒性

### 损失函数

$$\mathcal{L}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, \mathbf{x}_t \sim \mathcal{N}(\mathbf{x}_0, t^2 \mathbf{I})} \| \epsilon_\theta(\mathbf{x}_t, \mathbf{c}_\phi) - \epsilon_\theta^{\omega, \mathbf{c}_\text{null}}(\mathbf{x}_t, \mathbf{c}) \|$$

仅优化 sharpener 参数 $\phi$（2M），扩散模型 $\theta$ 完全冻结。训练使用与原模型相同的图文数据集。

## 实验关键数据

### 主实验：SD1.5 及变体定量对比

| 方法 | NFE | 可训练参数 | FID↓ | CLIP Score↑ | Aesthetic↑ | HPS v2.1↑ |
|------|-----|-----------|------|-------------|-----------|-----------|
| SD1.5 (ω=5, CFG) | 40 | - | 22.04 | 30.22 | 5.36 | 24.29 |
| SD1.5 (ω=1, 无引导) | 20 | - | 32.80 | 21.99 | 5.03 | 17.79 |
| Scaling (s=1.2) | 20 | - | 32.54 | 22.89 | 5.13 | 18.11 |
| GD (蒸馏) | 20 | 859M | 23.54 | 28.02 | 5.30 | 21.84 |
| PnP (蒸馏) | ≈28 | 361M | 26.57 | 27.72 | 5.39 | 23.17 |
| **DICE (本文)** | **20** | **2M** | **22.22** | **28.54** | **5.28** | **22.78** |

### 消融实验与跨模型泛化

| 实验设置 | FID↓ | CLIP Score↑ | 说明 |
|---------|------|-------------|------|
| DICE 完整 (SD1.5) | 22.22 | 28.54 | 基线 |
| 仅锐化 semantic embedding | >25 | <27 | 效果有限 |
| 仅锐化 padding embedding | ~23 | ~28 | 主要贡献来自 padding 分量 |
| DreamShaper DICE (NFE=20) | 30.80 | 29.40 | vs CFG (NFE=40) 30.35/30.50 |
| DreamShaper GD (NFE=20) | 32.53 | 28.48 | DICE 显著优于 GD |
| SDXL DICE | - | - | 跨架构有效 (UNet→DiT) |
| PixArt-α DICE | - | - | 跨编码器有效 (CLIP→T5) |

### 关键发现

- DICE 在 NFE=20（半计算量）下 FID 甚至优于 CFG 引导（NFE=40, FID=22.04）
- 在 DreamShaper 上 DICE (NFE=20) FID=30.80 vs CFG (NFE=40) FID=30.35，几乎无差距
- DrawBench 文本理解评估：DICE 23.32 vs CFG 23.83，差距极小
- 锐化模式分析：语义无关分量被主要放大，揭示 CFG 本质是增强 embedding 细粒度方向

## 亮点与洞察

- **揭示 CFG 的底层机制**：CFG ≈ text embedding sharpening，这一发现有独立理论价值
- **极致参数效率**：2M 参数 vs GD 859M 和 PnP 361M，减少 99.8%
- **2× 推理加速且几乎无损**：直接替换 CFG 的 drop-in 方案，无需修改扩散模型架构
- 首次支持 negative prompt 的 CFG 蒸馏方法，实用性更强
- AAAI Oral 认可，方法获学界高度评价

## 局限性

- 锐化强度 α 需要针对不同基础模型调整（同一基础模型的变体可复用）
- 未验证对视频扩散模型（如 SVD、CogVideoX）的适用性
- 极高 CFG scale（ω>15）下的效果未充分验证
- 训练仍需访问图文数据集和扩散模型的前向传播

## 相关工作

| 类别 | 代表方法 | 与 DICE 的关系 |
|------|---------|---------------|
| CFG 蒸馏 | GD, PnP | 蒸馏到模型参数 vs DICE 蒸馏到 embedding，后者参数量减少 99.8% |
| 采样加速 | LCM, DMD | 减少采样步数 vs DICE 减少每步计算，两者正交可叠加 |
| Guidance 替代 | PAG, APG | 仍需额外计算 vs DICE 完全消除额外前向传播 |
| Embedding 优化 | TextCraftor | 优化文本编码器权重 vs DICE 训练独立轻量 sharpener |

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 揭示 CFG 本质 + embedding sharpening 替代 CFG
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 SD1.5/SDXL/PixArt-α 全面验证
- 写作质量: ⭐⭐⭐⭐⭐ Oral 级别，机制分析深入透彻
- 价值: ⭐⭐⭐⭐⭐ 对所有使用 CFG 的扩散模型有即时实用价值
