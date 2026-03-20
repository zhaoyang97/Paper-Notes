# Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking

**会议**: NeurIPS 2025  
**arXiv**: [2505.18495](https://arxiv.org/abs/2505.18495)  
**代码**: 有（项目页面）  
**领域**: 图像生成 / LLM  
**关键词**: discrete diffusion, masked diffusion model, partial masking, intermediate states, text generation, image generation  

## 一句话总结
提出 Prime（Partial masking scheme），突破 Masked Diffusion Model 的二元状态（mask/unmask）限制，引入中间态（部分观测的 token 信息），减少冗余计算并实现更细粒度的去噪过程，在文本生成上 PPL 15.36 超越自回归模型（17.54）和标准 MDM（21.52），在图像生成上取得 CIFAR-10 FID 3.26。

## 背景与动机
Masked Diffusion Models（MDM）是离散数据生成的强大方法——逐步 unmask tokens 生成样本。但存在一个关键效率问题：连续采样步之间 token 序列常常不变（大部分 token 保持 masked 或 unmasked 状态不变），模型重复处理相同输入，导致大量冗余计算。

根本原因：每个 token 只有两种状态——完全 masked 或完全 unmasked，没有中间过渡，去噪过程是"跳跃式"的。

## 核心问题
如何让 MDM 的 token 具有**连续的中间状态**（部分观测/部分 masked），使去噪过程更细粒度、减少冗余计算、提升生成质量？

## 方法详解

### 关键设计
1. **部分掩码状态**: 扩展 token 的状态空间——从 {masked, unmasked} 扩展为 [masked, partially_observed, ..., unmasked] 的连续/半连续谱。中间态通过在 masked 和 unmasked 嵌入之间插值实现。

2. **变分训练目标**: 推导了适用于部分掩码状态的变分下界（ELBO），作为生成模型的训练目标。保持了 MDM 的理论基础。

3. **架构适配**: 引入简单的架构修改使模型能处理中间态输入——token 的嵌入不再是离散查找表的输出，而是连续的插值向量。

4. **细粒度去噪**: 在采样时，token 可以从 fully masked → partially revealed → more revealed → unmasked 逐步过渡，使模型在每一步都有新信息可处理，消除了连续步骤间输入不变的冗余。

### 训练策略
标准 MDM 训练流程 + 部分掩码的噪声调度。训练目标是变分下界。

## 实验关键数据

| 任务 | 方法 | 指标 |
|------|------|------|
| 文本 (OpenWebText) | 标准 MDM | PPL 21.52 |
| 文本 (OpenWebText) | 自回归模型 | PPL 17.54 |
| 文本 (OpenWebText) | AR+MDM 混合 | PPL 17.58 |
| 文本 (OpenWebText) | **Prime** | **PPL 15.36** |
| 图像 (CIFAR-10) | **Prime** | **FID 3.26** |
| 图像 (ImageNet-32) | **Prime** | **FID 6.98** |

**首个非自回归方法在文本生成上超越自回归模型**（PPL 15.36 vs 17.54）。

### 消融实验要点
- 部分掩码 vs 二元掩码：部分掩码显著提升生成质量
- 中间态的插值方式：线性插值 vs 可学习插值
- 采样步数与效率的权衡

## 亮点
- 打破了"离散扩散=二元状态"的范式限制
- 首次在文本生成上以非自回归方式超越自回归模型
- 理论基础扎实——变分训练目标有理论保证
- 跨模态通用——文本和图像上均有效

## 局限性 / 可改进方向
- 部分掩码的连续化使得原本离散的模型带有连续组件
- 在大规模数据（如 ImageNet-256）上的验证不足
- 与最新的 autoregressive+diffusion 混合方法（如 MAR）的对比
- 采样效率虽有改善但仍不如纯自回归模型的并行解码

## 与相关工作的对比
- **vs MDLM/SEDD（标准 MDM）**: Prime 通过引入中间态将 PPL 从 21.52 降至 15.36，提升幅度巨大
- **vs GPT-2（自回归）**: 首次以非 AR 方式超越 AR 模型（15.36 vs 17.54），里程碑式结果
- **vs ARGenSeg（同系列笔记）**: ARGenSeg 用 next-scale prediction 做离散生成分割；Prime 用部分掩码改善离散扩散

## 启发与关联
- 部分掩码的思想可迁移到 VQ-VAE 的 token 生成——在 code index 空间引入"soft index"
- 中间态概念可用于改善 MAE 的预训练——不是完全掩码/不掩码，而是部分遮挡
- 非 AR 超越 AR 的结果对 LLM 社区有重要意义——是否意味着自回归不是最优范式？

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 部分掩码突破二元状态限制，理论基础强
- 实验充分度: ⭐⭐⭐⭐ 文本+图像双模态，但大规模图像数据验证不足
- 写作质量: ⭐⭐⭐⭐ 概念清晰，理论推导完整
- 价值: ⭐⭐⭐⭐⭐ 首次非 AR 超越 AR 的文本生成结果，对生成模型社区意义重大
