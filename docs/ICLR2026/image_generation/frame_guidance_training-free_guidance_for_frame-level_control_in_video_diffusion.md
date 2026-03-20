# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2506.07177](https://arxiv.org/abs/2506.07177)  
**代码**: [https://frame-guidance-video.github.io/](https://frame-guidance-video.github.io/)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 无训练引导, 视频扩散模型, 帧级控制, 关键帧生成, 风格化视频  

## 一句话总结

提出 Frame Guidance，一种无需训练的帧级引导方法，通过 latent slicing（降低 60× 显存）和 Video Latent Optimization（VLO）两个核心组件，在不修改模型的情况下实现关键帧引导、风格化和循环视频等多种可控视频生成任务。

## 研究背景与动机

1. **可控视频生成需求增长**：随着视频扩散模型质量提升，用户对细粒度控制的需求日益迫切
2. **训练式方法不经济**：现有方法通常需要对大规模 VDM 进行微调，随着模型规模（如 Wan 14B）增长，微调成本越来越不可行
3. **无训练方法的通用性不足**：已有无训练方法（如 CamTrol、MotionClone）仅适用于特定任务，缺乏通用框架
4. **视频 CausalVAE 的显存瓶颈**：CausalVAE 的因果依赖要求解码全序列才能重建单帧，导致梯度计算显存超过 650GB
5. **现有引导策略对视频不适用**：图像领域的 time-travel trick 在视频早期步骤中会冲刷引导信号
6. **双重目标难以兼顾**：同时满足"模型无关 + 无训练"和"通用多任务"两个条件的方法尚属空白

## 方法详解

### 整体框架

Frame Guidance 在预训练 VDM 的推理过程中，对选定帧施加基于梯度的引导，通过三个核心组件实现高效可控生成：**Latent Slicing**（高效解码）、**VLO**（分阶段优化策略）、以及 **任务自适应损失设计**。

### 关键设计一：Latent Slicing

- **CausalVAE 的时间局部性发现**：实验表明 CausalVAE 虽设计为因果，但实际扰动仅影响相邻少数 latent（temporal locality）
- **切片解码**：重建第 $i$ 帧时只解码 3 个 latent 的窗口，而非整个序列
- **空间下采样**：对 latent 进行 $2\times$ 空间下采样后再解码计算损失
- **效果**：显存降低最多 **60×**，单 GPU 即可对 Wan-14B 等大模型进行引导

### 关键设计二：Video Latent Optimization (VLO)

- **核心洞察**：视频帧的全局布局在前几步去噪中就已确定；早期引导对时序一致性最为关键
- **分阶段策略**：
  - **早期阶段**（$t > t_E$）：确定性更新 $z_t \leftarrow z_t - \eta \nabla_{z_t} \mathcal{L}_e$，保留引导信号
  - **中间阶段**（$t_E \geq t > t_L$）：随机更新（加入 re-noising），修正累积误差
  - **后期阶段**（$t \leq t_L$）：无引导，自由细化细节

### 梯度传播的关键作用

- 引导梯度必须通过去噪网络 $v_\theta$ 传播，才能影响整个视频的时序一致性
- 仅对切片 latent 施加引导，但梯度通过网络传播到所有帧
- Shortcut-based 更新（跳过网络）仅影响被引导帧，导致时序断裂

### 多任务损失设计

| 任务 | 损失函数 |
|------|----------|
| 关键帧引导 | $\mathcal{L}_e = \sum_{i \in \mathcal{I}} \|x_*^i - x_{0\vert t}^i\|_2^2$ |
| 风格化 | $\mathcal{L}_e = -\sum_{i \in \mathcal{I}} \cos(\Psi(x_{\text{style}}), \Psi(x_{0\vert t}^i))$ |
| 循环视频 | $\mathcal{L}_e = \|x_{0\vert t}^1 - x_{0\vert t}^L\|_2^2$ |
| 通用条件（深度/边缘） | $\mathcal{L}_e = \sum_{i \in \mathcal{I}} \|\Psi(x_*^i) - \Psi(x_{0\vert t}^i)\|_2^2$ |

## 实验关键数据

### 关键帧引导（DAVIS 数据集）

| 方法 | 训练 | FID ↓ | FVD ↓ |
|------|------|-------|-------|
| CogX-I2V | ✓ | 60.36 | 890.1 |
| TRF (无训练) | ✓ | 62.07 | 923.1 |
| **Ours (CogX, I+F)** | ✓ | 57.62 | 613.4 |
| **Ours (CogX, I+M+F)** | ✓ | 55.60 | 577.1 |
| SVD-Interp (微调) | ✗ | 63.89 | 800.3 |
| CogX-Interp (微调) | ✗ | 46.59 | 506.0 |

### Pexels 数据集

| 方法 | FID ↓ | FVD ↓ |
|------|-------|-------|
| CogX-I2V | 74.98 | 1122.6 |
| **Ours (Wan-14B, I+M+F)** | 71.63 | 904.8 |
| **Ours (CogX, I+M+F)** | 68.97 | 989.3 |

**关键发现**：无训练的 Frame Guidance 在多数指标上超越训练式 SVD-Interp，仅略低于专门微调的 CogX-Interp。

## 亮点与洞察

1. **CausalVAE temporal locality 的发现**：虽然设计为因果，但实际呈现时间局部性——这一发现使显存降低 60× 成为可能
2. **VLO 的分阶段策略**：不同于图像的统一 time-travel，针对视频的时序特性设计确定性/随机性分阶段优化
3. **模型无关性**：在 CogVideoX、LTX-Video、Wan-14B 三种不同 VDM 上均有效
4. **极高的灵活性**：支持任意关键帧位置、多种条件信号、多种任务，无需为每个任务训练
5. **引导帧数可少**：仅引导少数帧即可通过网络梯度传播控制整个视频

## 局限性 / 可改进方向

1. 推理速度较慢（不超过基础模型 4× 的约束下工作），且引导步数和步长需手动调参
2. 关键帧引导不是像素级精确匹配，而是视觉相似性引导
3. 风格化依赖 CSD 等特定风格编码器的质量
4. 对于高动态场景（如快速运动、场景切换），早期步骤的布局确定可能不够充分
5. 尚未探索音频、文本等更多模态条件的引导

## 相关工作与启发

- **Universal Guidance (Bansal et al., 2024)**：图像领域 training-free guidance 的基础，本文将其拓展到视频
- **TRF (Feng et al., 2024)**：无训练 SVD 帧插值，但缺乏通用性；Frame Guidance 通过帧级损失设计实现更广泛的任务
- **CogX-Interp**：微调式关键帧插值方法，精度更高但需训练
- 启发：CausalVAE 的 temporal locality 特性可能被其他 training-free 方法（如编辑、修复）所利用

## 评分

- 新颖性: ⭐⭐⭐⭐ — Latent Slicing 和 VLO 都是针对视频场景的巧妙设计
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多任务、多数据集验证
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，分析深入，图示出色
- 价值: ⭐⭐⭐⭐⭐ — 在大模型时代的实用性极强，无训练方法的重要里程碑
