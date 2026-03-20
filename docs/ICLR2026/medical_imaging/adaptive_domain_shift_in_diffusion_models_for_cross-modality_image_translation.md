# Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation

**会议**: ICLR 2026  
**arXiv**: [2601.18623](https://arxiv.org/abs/2601.18623)  
**代码**: [https://github.com/LaplaceCenter/CDTSDE](https://github.com/LaplaceCenter/CDTSDE)  
**领域**: 医学图像 / 扩散模型  
**关键词**: 跨模态图像翻译, 扩散SDE, 域迁移调度, 空间自适应混合, 逆向SDE  

## 一句话总结
提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

## 研究背景与动机

1. **领域现状**：跨模态图像翻译（如MRI T1→T2、SAR→光学）已从GAN时代进入扩散模型时代，扩散方法在稳定性和生成质量上优于GAN。
2. **现有痛点**：现有扩散翻译方法普遍依赖源域→目标域之间的**固定线性插值** $d_t = \eta_t \hat{x}_0^{\text{src}} + (1-\eta_t) x_0$，这条直线路径会穿过两个模态流形之间的高能量区域，迫使采样器做大量偏离流形的校正。
3. **核心矛盾**：线性插值假设源-目标变换是全局均匀的，但真实跨模态差异在空间上高度异质——某些区域（如纹理差异大的边缘）需要更多校正，而均匀区域几乎不需要。
4. **本文要解决什么**：能否让域迁移调度本身学习一条"自适应弯曲"的路径，绕过高能量区域，从而减轻去噪负担并提高语义一致性？
5. **切入角度**：作者从路径能量泛函的几何视角出发，证明了在温和异质性条件下，逐像素自适应路径的能量严格低于任何全局调度路径（Theorem 1）。
6. **核心idea一句话**：将域迁移从"全局线性插值"升级为"逐像素、逐通道的可学习非线性混合场"，并将其嵌入扩散SDE的漂移项中。

## 方法详解

### 整体框架
CDTSDE（Cross-Domain Translation SDE）在VP扩散过程中引入自适应域混合场 $\Lambda_t \in (0,1)^{C \times H \times W}$。前向过程的边际分布以 $\sqrt{\bar\alpha_t} \cdot d_t$ 为均值（$d_t = \Lambda_t \odot \hat{x}_0^{\text{src}} + (1-\Lambda_t) \odot x_0$），逆向SDE的漂移项中包含显式的域迁移恢复力。输入是源模态图像，输出是目标模态翻译结果。

### 关键设计

1. **空间自适应域混合场（Adaptive Dynamic Domain Shift）**
   - 做什么：在每个逆向时步 $t$ 预测一个全分辨率的混合场 $\Lambda_t \in (0,1)^{C \times H \times W}$
   - 核心思路：用一个轻量卷积网络 $\mathcal{S}_\theta$ 接收基础线性步 $\lambda_t^{\text{lin}}$ 和位置编码 $\pi(p)$，输出空间调制信号 $h_{t,c}(p)$。通过零中心化变换 $g = 2h-1$ 和一个保端点的插值公式 $f_{t,c} = \lambda_t^{\text{lin}}[1 + g_{t,c}(1-\lambda_t^{\text{lin}})]$，再经calibrated logistic map压缩到 $(0,1)$，得到 $\Lambda_{t,c}(p)$
   - 设计动机：**Theorem 1**证明，在局部几何异质（不同像素有不同的最优混合比例）和非退化对比度条件下，$\inf_{\Lambda \in \mathcal{C}_{\text{pix}}} \mathcal{E}[d] < \inf_{\Lambda \in \mathcal{C}_{\text{glob}}} \mathcal{E}[d]$，即逐像素调度严格优于全局调度。这为空间自适应提供了理论支撑

2. **域感知前向/逆向SDE（Cross-Modal Diffusion Process）**
   - 做什么：将自适应混合场嵌入VP扩散的前向边际和逆向漂移项
   - 核心思路：前向边际 $q(x_t | x_0, \hat{x}_0^{\text{src}}) = \mathcal{N}(\sqrt{\bar\alpha_t} d_t, \sigma_t^2 I)$，增加的漂移 $\sqrt{\bar\alpha_t} \dot\Lambda(t) \odot (\hat{x}_0^{\text{src}} - x_0)$ 使前向均值追踪域混合路径。逆向SDE（Eq.9）包含三项力：标准漂移 $f(t)x_t$、域迁移恢复力、以及score函数
   - 设计动机：将域迁移物理直接编码到生成动力学中，使得即使采用大步长积分也能保持在流形上，因为每步更新本身就携带了域感知的校正方向

3. **精确解与一阶采样器（Exact Solution & First-order Sampler）**
   - 做什么：推导逆向SDE在变换坐标下的精确解（Proposition 1），并设计一阶数值采样器
   - 核心思路：引入坐标变换 $\Upsilon_t = \sqrt{\bar\alpha_t}(1-\Lambda_t)$，$y_t = x_t \oslash \Upsilon_t$，$\lambda_t = \sigma_t \oslash \Upsilon_t$，将逆向SDE化为可用variation-of-constants公式精确求解的形式。Proposition 1给出包含四项的精确解：(a) 缩放传播、(b) 数据预测积分、(c) 源图像恢复项、(d) 随机项
   - 设计动机：精确解保证了边际一致性，一阶采样器仅需5步即可达到 ~15dB PSNR，远快于需要1000步的BBDM等方法

4. **中间时步截断（Middle-point Truncation）**
   - 做什么：在采样起始时间 $t_1 < T$ 直接从 $x_{t_1} \sim \mathcal{N}(\sqrt{\bar\alpha_{t_1}} \hat{x}_0^{\text{src}}, \sigma_{t_1}^2 I)$ 初始化，跳过 $T - t_1$ 步
   - 设计动机：$t \geq t_1$ 后 $\Lambda_t = 1$，前向均值变为纯源图像中心的噪声过程，无需从纯噪声开始

### 训练策略
- 噪声预测模型 $\varepsilon_\theta$ 与域调度网络 $\mathcal{S}_\theta$ 联合训练
- UNet backbone + PyTorch Lightning混合精度
- 各任务训练步数适中：Sentinel 20K, IXI 10K, PSCDE 5K

## 实验关键数据

### 主实验
在三个跨模态翻译任务上与Pix2Pix、BBDM、ABridge、DBIM、DOSSR对比：

| 任务 | 指标 | CDTSDE | DOSSR(次优) | Pix2Pix |
|------|------|--------|------------|---------|
| Sentinel (SAR→Optical) | SSIM↑ | **0.382** | 0.360 | 0.230 |
| Sentinel | PSNR↑(dB) | **17.46** | 17.14 | 15.12 |
| IXI (T2→T1) | SSIM↑ | **0.825** | 0.800 | 0.710 |
| IXI (T2→T1) | PSNR↑(dB) | **24.33** | 24.13 | 22.24 |
| PSCDE (缺陷语义) | Dice↑ | **0.488** | 0.460 | 0.178 |
| PSCDE | Hausdorff↓ | **39.87** | 59.53 | 156.28 |

CDTSDE在几乎所有指标上居首，在效率方面仅需5个采样步（1.8s/图）达到15dB PSNR，比DOSSR（10步, 3.6s）快2x。

### 消融实验

| 调度类型 | Dice (PSCDE) | Hausdorff↓ | 说明 |
|---------|-------------|-----------|------|
| Linear (全局线性) | 0.46 | 59.5 | 固定 $\eta_t \cdot \mathbf{1}$ |
| Channel Non-linear | 0.46 | 43.0 | 逐通道非线性但空间均匀 |
| Dynamic (完整) | **0.49** | **39.8** | 空间+通道自适应 |

### 关键发现
- 从Linear→Dynamic，Dice提升6.1%，Hausdorff降低33%，说明空间自适应域调度的核心价值
- Channel Non-linear已能显著改善边界质量（Hausdorff 59.5→43.0），但区域重叠不变，空间维度的自适应提供了额外的overlap提升
- Bridge-based方法（BBDM、ABridge、DBIM）在高度异质的PSCDE任务上几乎完全失效（Dice<0.17），而CDTSDE和DOSSR因显式域迁移设计表现远好

## 亮点与洞察
- **理论驱动的设计**：Theorem 1从路径能量泛函角度严格证明了逐像素调度优于全局调度，这个理论结果不仅支撑了方法设计，还具有更广泛的启示——在任何需要学习两个分布间过渡路径的生成任务中，空间自适应调度都可能有益
- **精确解→高效采样**：通过坐标变换得到逆向SDE的精确解，实现5步高质量翻译，是理论到实践的典范
- **域迁移力嵌入漂移项**的设计让去噪模型从"全局对齐"降级为"局部残差校正"，大幅降低了学习难度

## 局限性 / 可改进方向
- 在低域差异场景（如IXI）改善幅度有限（SSIM从0.80→0.82），说明当模态差异小时额外的自适应调度并非必要
- 仅在配对数据上训练和评估，未探索非配对跨模态翻译
- GAN方法在感知质量（sharpness）上可能更好，CDTSDE可以考虑加入轻量感知/对抗损失
- 域调度网络 $\mathcal{S}_\theta$ 的容量和架构选择对性能的影响没有充分探讨
- 仅验证了256×256分辨率，高分辨率场景的计算开销和显存待评估

## 相关工作与启发
- **vs DOSSR**: 同为显式域迁移扩散方法，但DOSSR用固定线性调度，CDTSDE用可学习空间自适应调度，后者在PSCDE上Dice高3个点
- **vs BBDM/Bridge方法**: Bridge方法在配对数据间建布朗桥，但缺乏对域异质性的建模，在复杂翻译任务上严重退化
- **vs SDEdit**: SDEdit通过固定噪声水平控制翻译，无显式域迁移机制，在复杂跨模态场景下语义漂移严重

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将域迁移物理嵌入SDE漂移项+理论证明空间调度优越性，理论和方法都有创新
- 实验充分度: ⭐⭐⭐⭐ 三个不同难度任务+消融+效率分析完整，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，Fig.1的流形路径可视化直观，整体逻辑清晰
- 价值: ⭐⭐⭐⭐ 在医学图像和遥感领域有实际应用价值，自适应调度idea可迁移到其他条件生成任务
