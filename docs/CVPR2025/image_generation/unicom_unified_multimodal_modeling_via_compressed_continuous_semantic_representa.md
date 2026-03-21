<!-- 由 src/gen_stubs.py 自动生成 -->
# UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations

**会议**: CVPR2025
**arXiv**: [2603.10702](https://arxiv.org/abs/2603.10702)
**代码**: [Project Page](https://miazhao7708.github.io/UniComPage/)
**领域**: image_generation
**关键词**: unified multimodal model, continuous representation, semantic compression, transfusion, image generation

## 一句话总结
提出 UniCom，通过对 VLM 连续语义特征进行**通道维度压缩**（而非空间下采样），构建紧凑连续表示空间，用 Transfusion 架构统一多模态理解与生成，在统一模型中达到 SOTA 生成质量。

## 研究背景与动机
1. **统一模型的核心挑战**：当前统一多模态模型需要找到一个"统一 token"表示，同时支持理解和生成，但视觉表示设计一直是核心瓶颈
2. **离散化的信息损失**：基于向量量化的方法（如 VQ-VAE）不可避免地丢弃细粒度语义信息，导致理解任务性能次优
3. **混合编码器的表示分裂**：VAE latent + ViT 特征的混合方案导致理解和生成基于不同特征空间，限制了深度统一
4. **连续特征建模困难**：直接对高维连续 ViT 表示进行生成建模面临分布复杂、收敛慢、训练不稳定等问题
5. **流形假设**：高维 VLM 嵌入空间的有效信息实际位于低维子流形上，提示可通过学习压缩来发现该子流形
6. **实际需求**：需要在保留语义精度的同时简化数据分布，使生成建模变得可行

## 方法详解

### 整体框架
UniCom 将条件图像分布分解为两阶段：$P(\mathbf{x}|\mathbf{c}) = \int P(\tilde{\mathbf{z}}|\mathbf{c}) \cdot P(\mathbf{x}|\tilde{\mathbf{z}}) d\tilde{\mathbf{z}}$，其中 $\tilde{\mathbf{z}} \in \mathbb{R}^{N \times d}$（$d \ll D$）是压缩语义表示。框架包含三个核心组件：
- **语义压缩器**（Semantic Compressor）：将高维 VLM 特征映射到低维连续空间
- **生成先验模块**（Generative Prior Module）：在压缩空间中从文本条件采样
- **扩散解码器**（Diffusion Decoder）：从压缩表示重建像素图像

基于 Qwen-2.5-7B-Instruct 做理解，FLUX.1-dev 做生成解码，SigLIP2 做语义编码。

### 关键设计

**1. 基于注意力的语义压缩器**
- 实现为浅层轻量 Transformer 模块，而非简单 MLP
- 上下文感知的映射保留了 patch 间的长程语义关系
- 关键发现：**通道维度压缩远优于空间下采样**——压缩通道 18× (1152→64) 几乎无损，但减少 token 数量会导致细节模糊

**2. 压缩器与解码器联合优化**
- 压缩器参数 $\phi$ 与扩散解码器参数 $\psi$ 联合训练
- 使用重建目标：$\mathcal{L}_{\text{recon}} = \mathcal{L}_{\text{flow}}(\mathbf{x}, \hat{\mathbf{x}}) + \lambda \cdot \mathcal{L}_{\text{perc}}(\mathbf{x}, \hat{\mathbf{x}})$
- 联合训练迫使压缩器形成信息瓶颈，保留生成有用的信号

**3. 两种预测路径对比**
- **Pathway I (Transfusion)**：统一 Transformer 处理文本 + 图像混合序列，文本用因果 mask，图像用双向注意力，端到端 flow matching 训练
- **Pathway II (Query-Guided)**：用 MetaQuery 从冻结 MLLM 提取条件信号，再通过连接器投影到 flow matching 解码器
- 对比结论：Transfusion 收敛更快，编辑任务一致性更好

### 损失函数
- 重建阶段：Flow matching loss + 感知损失（LPIPS）
- 生成阶段：$\mathcal{L}_{\text{FM}} = \mathbb{E}[\|\mathbf{v}_t - \mathbf{v}_\theta(\tilde{\mathbf{z}}_t, t; \mathbf{c})\|_2^2]$，标准 flow matching 目标
- 理解阶段：交叉熵损失 $\mathcal{L}_{ce}$
- 四阶段渐进训练：对齐→预训练→持续训练→监督微调

## 实验关键数据

### 主实验：图像重建（ImageNet val）
| 方法 | rFID↓ | PSNR↑ | SSIM↑ |
|------|-------|-------|-------|
| FLUX.1-dev VAE | 0.06 | 33.65 | 0.93 |
| UniCom (d1152, 未压缩) | 0.38 | 22.60 | 0.61 |
| UniCom (d64, 18×压缩) | 0.42 | 22.28 | 0.61 |
| UniTok | 0.38 | - | - |
| SD-VAE | 1.06 | 28.62 | 0.86 |

关键发现：通道从 1152 压缩到 64（18×），rFID 仅从 0.38 升至 0.42，证明通道压缩几乎无损。

### 主实验：文本到图像生成（GenEval）
| 模型 | Overall |
|------|---------|
| UniCom | **0.91** (SOTA among unified) |
| FLUX.1-Dev | 0.82 |
| Janus-Pro | 0.80 |
| Show-o2 | 0.76 |

### 消融实验
- **通道 vs 空间压缩**：相同压缩比下，通道压缩在重建和生成上均显著优于空间下采样
- **注意力 vs MLP 压缩器**：注意力压缩器在语义保持和重建质量上优于 MLP
- **Transfusion vs Query-Guided**：Transfusion 收敛更快、编辑一致性更高

### 关键发现
- 语义连续表示可以在通道维度大幅压缩而几乎不损失信息
- 统一模型在文本渲染和复杂编辑上表现出色，无需依赖 VAE 做身份保持
- GenEval Overall 0.91 超越所有统一模型，包括 OmniGen2 (0.86) 和 Mogao (0.89)

## 亮点与洞察
1. **通道压缩 >> 空间压缩**的发现具有普适指导意义。直觉是空间 token 包含位置信息和局部细节，而通道维度存在大量冗余
2. **连续语义表示 + 压缩**的方案优雅解决了离散化损失和高维建模困难的两难
3. 系统性对比 Transfusion vs Query-Guided 两种范式，提供了有价值的设计指南
4. 不依赖 VAE 即可实现高质量图像编辑中的身份保持，说明压缩语义表示包含足够信息

## 局限性
1. 重建质量（PSNR/SSIM）仍明显低于专用 VAE（如 FLUX VAE），说明语义压缩不擅长保留像素级精确度
2. 四阶段训练流程复杂，训练成本较高
3. 理解任务用未压缩特征、编辑任务用压缩特征的不对称设计增加了系统复杂度
4. 仅在 7B 规模验证，更大规模的 scaling 行为未知

## 相关工作与启发
- **vs Janus-Pro/Show-o2**：UniCom 避免了离散化损失，生成质量显著领先
- **vs VUGEN**：同样使用连续表示但 UniCom 用注意力压缩器替代 MLP，保留结构语义更好
- **vs Transfusion**：在其架构基础上引入语义压缩，大幅改善训练稳定性和收敛速度
- **启发**：通道压缩策略可推广到其他需要特征瓶颈的场景（如视频理解、3D 生成）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 通道压缩 vs 空间压缩的系统性分析和注意力语义压缩器设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 重建/生成/编辑多任务评估，消融充分，但缺少理解任务定量对比
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，两种路径的对比设计严谨
- 价值: ⭐⭐⭐⭐⭐ — 统一模型 SOTA 生成质量，通道压缩发现对后续工作有重要指导意义
