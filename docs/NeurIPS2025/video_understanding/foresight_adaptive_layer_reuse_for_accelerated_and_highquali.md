# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation

## 基本信息
- **arXiv**: 2506.00329
- **会议**: NeurIPS 2025
- **作者**: Muhammad Adnan, Nithesh Kurella, Akhil Arunkumar, Prashant J. Nair
- **机构**: University of British Columbia, d-Matrix
- **代码**: https://github.com/STAR-Laboratory/foresight

## 一句话总结
提出 Foresight，一种训练无关的自适应层复用框架，通过动态 MSE 阈值决策在 DiT 去噪过程中哪些层可复用缓存、哪些需重新计算，在 OpenSora/Latte/CogVideoX 上实现最高 1.63× 端到端加速且保持视频质量。

## 背景与动机
DiT 视频生成的推理瓶颈来自两方面：(1) 空时注意力的 $\mathcal{O}(L^2)$ 复杂度随分辨率和帧数增长；(2) 数十步去噪过程的累积计算。

现有特征缓存方法（Static, PAB, Δ-DiT, T-GATE）采用**静态复用策略**——固定间隔、所有层统一处理。但作者发现复用潜力在以下三个维度高度变化：
1. **层间差异**：后期层特征变化更大，不适合粗暴复用
2. **Prompt 依赖**：快速场景变化的 prompt 复用潜力低
3. **配置敏感**：分辨率、帧数、去噪调度改变复用模式

## 核心问题
如何自适应地决策每一步每一层是否复用缓存，实现速度与质量的最优平衡？

## 方法详解

### 1. Warmup Phase
前 $W$ 步（默认 15%）正常计算所有层，让特征稳定后：
- 初始化缓存 $\mathcal{C}$ 
- 建立每层的自适应复用阈值 $\lambda$：
$$\lambda_{\mathbf{x}}^l = \sum_{t=W-2}^{W} \frac{1}{10^{W-t}} \left(\frac{1}{P}\sum_{i=1}^P (x_i^l(t) - x_i^l(t-1))^2\right)$$
用几何加权的最后三步 MSE，阈值因层、prompt、分辨率而异。

### 2. Reuse Phase
交替进行复用（$N$ 步）和重计算（每 $R$ 步）：

重计算步更新复用指标 $\delta$：
$$\delta_{\mathbf{x}}^l(t) = \frac{1}{P}\sum_{i=1}^P (x_i^l(t) - \mathcal{C}_i^l(t-1))^2$$

下一步按阈值决策：
$$\mathbf{x}_{t+1}^l = \begin{cases} \mathcal{C}(\mathbf{x}_t^l), & \text{if } \delta_{\mathbf{x}}^l(t) \leq \gamma \lambda_{\mathbf{x}}^l \\ \text{Compute}, & \text{otherwise} \end{cases}$$

缩放因子 $\gamma \in (0, 2]$ 控制速度-质量平衡。

### 3. 关键设计选择
- **粗粒度复用**：复用整个 DiT block（而非 PAB 的细粒度 attention/MLP 分离），缓存开销降低 3×
- **逐层独立决策**：后期层更频繁重计算，前期层大量复用
- **收敛性保证**：证明自适应复用的误差有界且可控：$\|\hat{\mathbf{x}}_t - \mathbf{x}_t^*\| \leq \varepsilon_{tot}/(1-\rho)$

## 实验关键数据

### VBench Benchmark (550 prompts)
| 模型 | 方法 | VBench Acc | PSNR↑ | SSIM↑ | 加速比 |
|---|---|---|---|---|---|
| OpenSora | PAB | 75.32 | 25.67 | 0.85 | 1.26× |
| | **Foresight** (N=1,R=2) | **75.90** | **29.67** | **0.90** | **1.28×** |
| | **Foresight** (N=2,R=3) | **75.62** | 27.49 | 0.87 | **1.44×** |
| CogVideoX | PAB | 77.89 | 29.04 | 0.91 | 1.37× |
| | **Foresight** (N=1,R=2) | **77.94** | **34.75** | **0.95** | **1.46×** |
| | **Foresight** (N=2,R=3) | **77.84** | 28.45 | 0.87 | **1.63×** |

### 扩展到 HunyuanVideo/Wan-2.1
- HunyuanVideo：Foresight 达 1.62× 加速，PSNR 41.79 远超 TeaCache
- Wan-2.1：Foresight 达 2.23× 加速

### 消融实验
- $\gamma=0.25$：PSNR 38.09（比 PAB 高 +9.97），延迟仅增加 0.62s
- $\gamma=2.0$：PSNR 29.51，最大加速
- 最佳 warmup：15%
- 缓存开销：Foresight 仅需 $2L \cdot H \cdot W \cdot F$，比 PAB 的 $6L \cdot H \cdot W \cdot F$ 少 3×

## 亮点
1. **自适应而非静态**：每层每步独立决策，适应 prompt/分辨率/调度的变化
2. **训练无关，即插即用**：不改架构，不需额外训练
3. **理论保证**：证明有界误差和收敛性
4. **广泛验证**：5 个模型（OpenSora, Latte, CogVideoX, HunyuanVideo, Wan-2.1）+ FLUX T2I
5. **质量优于速度优先**：在同等加速下质量全面超越静态方法

## 局限性
1. 加速比受限于复用窗口 $N$ 和 warmup $W$ 的配置
2. 目前采用粗粒度（block 级）复用，细粒度可能进一步提升
3. 自适应阈值依赖 warmup 阶段的 MSE 估计质量
4. 1.63× 加速幅度相对有限（vs. 步数压缩或蒸馏的 10-50×）

## 与相关工作的对比
- **vs. PAB**：PAB 按经验固定不同注意力类型的 broadcast 范围，Foresight 按数据驱动动态决策
- **vs. TeaCache**：TeaCache 利用 timestep embedding 的变化量做缓存判断，Foresight 用特征 MSE，后者更精确
- **vs. Δ-DiT**：Δ-DiT 缓存残差偏移量而非完整特征，且仍是静态方案
- **vs. 步数压缩/蒸馏**：Foresight 与这些方法正交，可组合使用

## 启发与关联
- **与 InfinityStar 的互补**：InfinityStar 将 AR 推理控制在极少步数，Foresight 针对扩散模型减少每步计算——两者分别代表 AR 和 Diffusion 的效率优化方向
- **自适应粒度的未来**：从 block 级复用扩展到 attention head 级或 token 级复用是自然方向
- **系统层面优化**：Foresight 的设计考虑了 FlashAttention 兼容性和 GPU VRAM，有工程落地性

## 评分
- 新颖性：★★★☆☆ — 自适应缓存的思路不算全新，但阈值设计和分析有价值
- 技术深度：★★★★☆ — 收敛性证明和系统性分析扎实
- 实验完整度：★★★★★ — 5 模型 × 多 benchmark × 多配置 × 消融
- 写作质量：★★★★☆ — 清晰，但略冗长
