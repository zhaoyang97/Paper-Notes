# Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2502.01051](https://arxiv.org/abs/2502.01051)  
**代码**: [https://github.com/Kwai-Kolors/LPO](https://github.com/Kwai-Kolors/LPO)  
**领域**: 图像生成 / 偏好优化  
**关键词**: diffusion model, preference optimization, reward model, latent space, step-level, noise-aware, DPO  
**机构**: 中科院自动化所 + 快手  

## 一句话总结
提出 Latent Reward Model (LRM) 和 Latent Preference Optimization (LPO)，将预训练扩散模型本身复用为噪声感知的潜空间奖励模型，在噪声潜在空间直接进行步级偏好优化，相比 Diffusion-DPO 实现 10-28× 训练加速，相比 SPO 实现 2.5-3.5× 加速。

## 背景与动机

### 现有方法的三大痛点
现有步级偏好优化方法（如 SPO）使用 VLM（CLIP 等）作为像素级奖励模型 (PRM)，存在三个关键问题：

1. **变换复杂**：每个时间步 t 都需要额外的扩散前向（x_t → x̂₀,t）+ VAE 解码（x̂₀,t → I_t）才能得到像素图像喂给 VLM，采样时间是 LRM 的 6 倍
2. **高噪声不兼容**：大时间步（高噪声）下预测的像素图像严重模糊，与 VLM 训练数据（清晰图像）分布严重偏移，导致 PRM 在大时间步预测不可靠
3. **时间步不敏感**：PRM 不以时间步为输入，无法理解不同去噪阶段对图像评估的影响差异

### 核心洞察
**预训练扩散模型天然满足步级奖励建模的所有需求**：
- 具有文本-图像对齐能力（大规模文图预训练）
- 能直接处理噪声潜在图像 x_t，无需额外解码
- 高噪声兼容（预训练就是处理各种噪声水平）
- 对去噪时间步天然敏感

## 方法详解

### 1. Latent Reward Model (LRM) 架构

LRM 复用扩散模型的 U-Net（或 DiT）和文本编码器组件：

- **文本特征**：文本编码器提取 prompt 特征 f_p，取 EOS token 特征 f_eos 经文本投影层得到最终文本特征 T ∈ ℝ^{1×n_d}
- **视觉特征**：噪声潜在图像 x_t 通过 U-Net，spatial 维度平均池化后得到多尺度 down-block 特征 V_down 和 mid-block 特征 V_mid
- **Visual Feature Enhancement (VFE)**：受 Classifier-Free Guidance 启发，额外提取无文本条件的 V_mid_uncond，增强视觉特征的文本相关性：V_enh = V_mid + (gs-1)·(V_mid - V_mid_uncond)，gs=7.5
- **偏好分数**：V_enh 与 V_down 拼接后投影得到视觉特征 V，最终分数 S(p, x_t) = τ · l₂(V) · l₂(T)（类 CLIP 点积）

VFE 模块的效果：gs 越大，文本对齐相关性越强（CLIP-Corr 提升），美学相关性适度下降（Aes-Corr），gs=7.5 达到最佳平衡。

### 2. Multi-Preference Consistent Filtering (MPCF)

**问题**：训练数据中约一半 winning image 在美学上不如 losing image，约 40% 在 CLIP/VQA 分数上更低。加噪后偏好排序可能翻转。

**方案**：用美学分数 S_A、CLIP 分数 S_C、VQA 分数 S_V 三个维度过滤 Pick-a-Pic v1 数据集：
- 策略1（最严格）：G_A≥0, G_C≥0, G_V≥0 → 101K 对，但 LRM 过拟合美学
- **策略2（最终采用）**：G_A≥-0.5, G_C≥0, G_V≥0 → 169K 对，美学与对齐平衡最好
- 策略3（最宽松）：G_A≥-1, G_C≥0, G_V≥0 → 202K 对，LRM 忽视美学

### 3. Latent Preference Optimization (LPO)

**采样**：每个时间步 t，从同一 x_{t+1} 采样 K=4 个 x_t^i，LRM 直接在噪声潜空间预测偏好分数 S_t^i，选最高分为 x_t^w、最低分为 x_t^l（需 SoftMax 归一化后差值超过阈值 th_t）。

**训练目标**：与 SPO 相同的步级 DPO 损失（公式6），但全部在噪声潜空间完成，无需 x̂₀,t 预测和 VAE 解码。

**优化时间步覆盖 t∈[0,950]**：SPO 因 SPM 在高噪声下不准确，只能覆盖 t∈[0,750]。LRM 作为噪声感知模型，可覆盖全部去噪过程。消融实验表明 t∈[750,950] 的高噪声范围对偏好优化至关重要。

**动态阈值**：σ_t 随 t 减小而降低，固定阈值效果差。采用线性映射 th_t 到 [th_min, th_max]=[0.35,0.5]（SD1.5）/ [0.45,0.6]（SDXL），小时间步用低阈值。

**同构/异构优化**：LRM 与被优化模型 DMO 可以是同架构（同构）或不同架构（异构），唯一约束是共享相同 VAE 编码器。实验证明用 SD1.5 的 LRM 微调 SD2.1（相同 VAE）效果显著，但微调 SDXL（不同 VAE）无效。

## 实验关键数据

### 主实验（SD1.5 / SDXL）
| 指标 | SD1.5 原始 | SPO | LPO | SDXL 原始 | SPO | LPO |
|------|-----------|-----|-----|-----------|-----|-----|
| PickScore | 20.56 | 21.22 | **21.69** | 21.65 | 22.70 | **22.86** |
| ImageReward | 0.008 | 0.168 | **0.659** | 0.478 | 0.995 | **1.217** |
| Aesthetic | 5.468 | 5.927 | **5.945** | 5.920 | 6.343 | **6.360** |
| GenEval(20s) | 42.56 | 40.46 | **48.39** | 49.40 | 50.52 | **59.27** |

LPO 在 SDXL 上甚至略超使用内部高质量数据集的 InterComp。

### T2I-CompBench++（文图对齐细粒度）
LPO 在颜色、形状、纹理、空间关系、计数等所有维度上全面超越 SPO 和 Diffusion-DPO。

### 训练效率
| 方法 | SD1.5 总训练 | SDXL 总训练 |
|------|-------------|-------------|
| Diffusion-DPO | 240 A100h | 2560 A100h |
| SPO | 80 A100h | 234 A100h |
| **LPO** | **23 A100h** | **92 A100h** |

单步采样：LRM 0.039s vs SPM 0.243s（6.2× 加速），因为省去了 x̂₀,t 预测和 VAE 解码。

### 关键消融
- **时间步范围**：[0,950] 全范围最优；仅 [750,950]（高噪声段）就能达到接近全范围的性能，证明高噪声步级优化至关重要
- **MPCF 策略**：不用 MPCF 的 LPO 仍优于 SPO，说明 LRM 本身优势显著；加 MPCF 进一步提升
- **动态阈值**：优于所有固定阈值设置，[0.35,0.5] 最优

## 亮点与贡献
1. **洞察原创**："扩散模型本身就是最好的步级奖励模型"——将扩散模型从被优化对象变为奖励信号来源，首次在噪声潜空间做奖励建模
2. **效率大幅提升**：省去像素空间往返计算，23 A100h 即可完成 SD1.5 全流程优化
3. **高噪声覆盖**：LRM 能在 t∈[750,950] 可靠预测偏好，突破 PRM 的高噪声限制
4. **VFE 模块**：借鉴 CFG 思想增强视觉特征的文本相关性，简洁有效
5. **异构优化**：低配模型 LRM 可微调高配模型（共享 VAE 即可）

## 局限性
- LRM 偏好预测准确度受限于扩散模型自身的表示质量
- 同构优化中 LRM 与 DMO 共享参数可能引入偏差
- 异构优化要求 VAE 编码器相同，限制跨架构泛化
- 仅验证了图像生成，未扩展到视频扩散模型
- MPCF 依赖外部评分器（Aesthetic Score、CLIP Score），引入额外计算

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型复用为噪声感知奖励模型
- 实验充分度: ⭐⭐⭐⭐⭐ SD1.5/SDXL/SD3 + 多维度评估 + 详尽消融 + 异构优化
- 写作质量: ⭐⭐⭐⭐ 动机清晰、图示直观
- 价值: ⭐⭐⭐⭐⭐ 10-28× 加速的实用方案，代码开源
