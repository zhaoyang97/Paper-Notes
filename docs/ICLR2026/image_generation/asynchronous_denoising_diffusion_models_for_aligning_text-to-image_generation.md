# Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation

**会议**: ICLR 2026  
**arXiv**: [2510.04504](https://arxiv.org/abs/2510.04504)  
**代码**: [https://github.com/hu-zijing/AsynDM](https://github.com/hu-zijing/AsynDM) (有)  
**领域**: 扩散模型 / 文图对齐  
**关键词**: 异步去噪, 像素级时间步, 文图对齐, cross-attention mask, plug-and-play  

## 一句话总结
AsynDM 通过为不同像素分配不同的时间步调度（prompt 相关区域去噪更慢），使其能利用更清晰的上下文参考，从而在不需要微调的情况下显著提升文图生成的语义对齐。

## 研究背景与动机
1. **领域现状**：扩散模型在文图生成中取得了优异的多样性和保真度，但文图对齐（alignment）仍是显著痛点——生成的图像经常在文字、颜色、数量等方面与 prompt 不一致
2. **现有痛点**：
   - 现有方法要么需要微调（RL-based alignment），要么在推理时修改 CFG 或中间噪声图像
   - 这些方法都没有触及同步去噪这一根本机制
3. **核心矛盾**：同步去噪中所有像素按相同时间步演进，prompt 相关区域只能参考同等噪声水平的其他区域作为上下文——但这些参考区域本身也是模糊的，无法提供清晰的语义引导
4. **本文要解决什么**：让 prompt 相关区域（如目标对象）在去噪过程中获得更清晰的上下文参考，以改善最终图像与 prompt 的语义对齐
5. **切入角度**：观察到图像中不同区域对去噪精细度的需求不同——背景约束少可以快速去噪，而 prompt 相关对象需要更精细的渐进式去噪
6. **核心 idea 一句话**：让 prompt 无关区域先变清晰作为更好的上下文参考，prompt 相关区域慢慢去噪以更好地聚焦 prompt 语义

## 方法详解

### 整体框架
AsynDM 是一个 plug-and-play、无需微调的框架。核心思想是将标量时间步 $t$ 扩展为像素级时间步张量 $\mathbf{t}_i \in \mathbb{R}^{h \times w}$，不同像素可以处于不同的噪声水平。通过 cross-attention 提取 prompt 相关区域的 mask，动态调制不同区域的去噪速度。

### 关键设计

1. **像素级时间步分配 (Pixel-Level Timestep Allocation)**:
   - 做什么：将标量时间步扩展为空间张量，每个像素有独立的时间步
   - 核心思路：在扩散模型中，时间步信息通过 pixel-wise 的方式嵌入特征（在注意力模块之外），而非直接注入注意力计算——这意味着不同像素天然可以关联不同时间步。DDPM 公式扩展为 $p_\theta(\mathbf{x}_{i+1}|\mathbf{x}_i, \mathbf{c}) = \mathcal{N}(\mathbf{x}_{i+1} | \mu_\theta(\mathbf{x}_i, \mathbf{t}_i, \mathbf{c}), \sigma_i^2 \mathbf{I})$，其中 $\alpha_{\mathbf{t}_i}$、$\beta_{\mathbf{t}_i}$ 是逐元素索引
   - 设计动机：保持了马尔科夫性质，状态从 $\mathbf{x}_t$ 扩展为 $(\mathbf{x}_i, \mathbf{t}_i)$

2. **凹函数时间步调度 (Concave Timestep Scheduling)**:
   - 做什么：prompt 相关区域按凹函数调度去噪（更慢），其他区域按线性调度（更快）
   - 核心思路：使用二次函数 $f(i) = T - \frac{1}{T}i^2$ 作为调度函数。Proposition 1 证明了位于凹函数与线性函数之间区域的任何点，都可以通过适当平移的凹函数到达 $t=0$
   - 设计动机：凹函数使得目标区域在早期几乎不去噪，而在后期加速去噪——这样在中间阶段，目标区域仍处于高噪声状态但能看到已经较清晰的背景区域，从而获得更好的上下文指导

3. **Mask 引导的异步去噪 (Mask-Guided Asynchronous Denoising)**:
   - 做什么：在每个去噪步从 cross-attention map 中提取 prompt 相关区域 mask，动态调制时间步
   - 核心思路：对 prompt 中每个目标 token $o$，取其 cross-attention map $A^o$，以均值为阈值二值化，再对所有目标 token 的 mask 做 OR 运算得到最终 mask $M = \bigvee_{o \in \mathcal{O}_\mathbf{c}} \mathbf{1}[A^o > A^o_{\text{mean}}]$
   - 设计动机：cross-attention map 天然编码了图像区域与文本 token 的对应关系，随着去噪推进 mask 越来越精确地定位目标形状

### 损失函数 / 训练策略
- **无需训练**：AsynDM 直接在预训练扩散模型上使用，只修改推理过程
- 兼容 DDPM、DDIM 等多种采样器
- 时间步编码独立处理后以 per-pixel 方式注入

## 实验关键数据

### 主实验 — 4 个 prompt 集上的对齐性能（SD 2.1）

| 方法 | BERTScore↑ | CLIPScore↑ | ImageReward↑ | QwenScore↑ |
|------|-----------|-----------|-------------|-----------|
| DM (baseline) | 0.6353 | 0.3685 | 0.7543 | 4.94 |
| Z-Sampling | 0.6353 | 0.3708 | 0.8283 | 5.02 |
| SEG | 0.6309 | 0.3605 | 0.6493 | 4.76 |
| S-CFG | 0.6383 | 0.3716 | 0.8653 | 5.04 |
| CFG++ | 0.6249 | 0.3565 | 0.3284 | 4.45 |
| **AsynDM** | **0.6414** | **0.3750** | **0.9219** | **5.52** |

（以 Animal Activity 为例，其他 3 个集上趋势一致）

### 消融实验 — 调度函数对比

| 配置 | BERTScore | ImageReward |
|------|-----------|-------------|
| 线性调度（baseline DM）| 0.6353 | 0.7543 |
| 全局凹函数（DMconcave）| 0.6381 | 0.8544 |
| **异步（AsynDM）** | **0.6414** | **0.9219** |

### 关键发现
- **AsynDM 在所有 4 个 prompt 集、4 个指标上均为最优**，且是唯一不需要微调的方法
- **QwenScore 提升最显著**：Animal Activity 上 +0.58（从 4.94 到 5.52），说明 VLM 评测认为对齐改善很大
- **SEG 和 CFG++ 反而损害对齐**：说明简单修改 guidance 不一定有效
- **mask 质量随去噪推进而提升**：早期 mask 粗糙但足够定位大致区域，后期精确捕捉物体形状

## 亮点与洞察
- **重新思考同步去噪**：之前的工作几乎都默认所有像素同步去噪，本文首次指出这是对齐问题的根源之一并提出解决方案——视角新颖
- **plug-and-play 实用性强**：不需要训练、不需要额外模型、兼容 UNet 和 DiT 架构，易于部署
- **凹函数调度的数学优雅性**：Proposition 1 保证了任意时刻被选为目标的区域都能通过平移的凹函数最终到达 t=0，避免了复杂的状态管理

## 局限性 / 可改进方向
- 依赖 cross-attention map 的质量来提取 mask，如果 prompt 中的实体在 attention 中未被正确定位则无效
- 二次函数 $f(i) = T - i^2/T$ 是手选的，不同 prompt 可能需要不同的调度强度
- 额外的像素级时间步编码会增加一些计算开销（虽然论文说可忽略）
- 对 prompt 中隐含的抽象概念（如风格、情绪）可能不如对具体物体有效

## 相关工作与启发
- **vs Z-Sampling**：Z-Sampling 引入 zigzag 步骤改善对齐，但所有像素仍同步；AsynDM 从像素级分化入手
- **vs SEG/S-CFG/CFG++**：这些方法修改 guidance 策略，AsynDM 修改时间步调度——正交的改进方向，理论上可以组合
- **vs Attend-and-Excite**：A&E 需要优化中间 latent，AsynDM 只修改时间步编码，更轻量

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 像素级异步去噪是全新的视角，重新定义了扩散模型的 MDP 状态
- 实验充分度: ⭐⭐⭐⭐ 4 个 prompt 集 + 4 个指标 + 多个 baseline + 消融，但缺少人类评测
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述极清晰，图示直观，数学推导优雅
- 价值: ⭐⭐⭐⭐ 提升对齐性能显著且实用，但场景受限于具体物体的对齐
