# Balanced Conic Rectified Flow

**会议**: NeurIPS 2025  
**arXiv**: [2510.25229](https://arxiv.org/abs/2510.25229)  
**代码**: [项目页面](https://grainsack.github.io/BC_rectified_flow_project_page/)  
**领域**: 图像生成 / 流匹配  
**关键词**: rectified flow, reflow, conic interpolation, Slerp, distribution drift

## 一句话总结

针对 k-rectified flow 中 reflow 步骤导致的分布漂移问题，提出 conic reflow：利用真实图像的反演噪声及其 Slerp 扰动构成锥形监督轨迹，大幅减少所需 fake pair 数量的同时获得更优的生成质量和更直的 ODE 路径。

## 研究背景与动机

1. **领域现状**：Rectified Flow 通过学习噪声到数据的 ODE 速度场实现高效生成，k-rectified flow 通过反复 reflow 将轨迹拉直，从而支持少步甚至单步生成。当前 SOTA 模型（Flux、SD3、AuraFlow）都采用 1-rectified flow + ~30 NFE 的方案。

2. **现有痛点**：
   - Reflow 需要大量 fake pair（原始方法使用 4M 对），生成代价高昂
   - Fake pair 来自不完美的模型，本身就偏离真实分布，导致 reflow 训练的监督信号存在系统性偏差
   - 多轮 reflow 会使误差不断累积，模型逐步远离真实数据分布

3. **核心矛盾**：Reflow 旨在拉直轨迹以支持少步生成，但其依赖的 fake pair 监督本身会引入分布漂移（distribution drift），导致生成质量在 full-step 下反而下降，形成"拉直轨迹"与"保持分布一致性"之间的矛盾。

4. **本文要解决什么？**：揭示 reflow 引起的分布漂移现象，并设计一种新的 reflow 策略使模型在拉直轨迹的同时保持对真实数据分布的忠实度。

5. **切入角度**：通过真实图像的重建误差（reconstruction error）定量揭示漂移——fake 图像的重建误差远低于 real 图像，说明模型过拟合了 fake 分布而偏离了 real 分布。进而提出用真实图像反演噪声 + Slerp 扰动构建"锥形"监督作为纠偏手段。

6. **核心 idea 一句话**：用真实图像的反演噪声及其 Slerp 邻域构成锥形监督轨迹，交替训练 real pair 和 fake pair，既纠正分布漂移又保证轨迹平直。

## 方法详解

### 整体框架

Balanced Conic Rectified Flow 包含三个关键组件：
1. **Real pair**：用已训练模型对真实图像 $X_1$ 做反演得到噪声 $Z_{0,R} = v^{-1}(X_1)$，构成 $(Z_{0,R}, X_1)$
2. **Conic reflow**：对反演噪声施加 Slerp 扰动，将监督信号从单条轨迹扩展到锥形邻域
3. **Balanced training**：交替使用 real pair（conic reflow）和 fake pair（原始 reflow），平衡分布忠实度与域覆盖

### 关键设计

1. **Real Pair 构造**：
   - 核心思路：不再依赖 $(Z_0, v(Z_0))$ 这种 fake pair，而是用真实图像 $X_1$ 及其反演 $Z_{0,R} = v^{-1}(X_1)$ 组成 real pair
   - 动机：fake pair 的终点 $Z_1 = v(Z_0)$ 偏离真实分布 $\pi_1$，而 real pair 的终点就是真实数据，能直接锚定目标分布
   - 反演过程利用 ODE 的确定性性质，无需引入随机性，实现简洁

2. **Conic Reflow（Slerp 扰动）**：
   - 核心思路：对反演噪声 $Z_{0,R}$ 施加 Slerp 插值扰动，将监督扩展到邻域
   - Slerp 公式：$$\text{Slerp}(Z_{0,R}, \epsilon, \zeta) = \frac{\sin((1-\zeta)\phi)}{\sin(\phi)} Z_{0,R} + \frac{\sin(\zeta\phi)}{\sin(\phi)} \epsilon$$
   - 其中 $\phi = \arccos(Z_{0,R} \cdot \epsilon)$，$\epsilon \sim \mathcal{N}(0, I)$，$\zeta$ 为插值比例
   - Conic 插值：$$\text{Conic}(X_1, \epsilon, \zeta, t) = t X_1 + (1-t) \cdot \text{Slerp}(Z_{0,R}, \epsilon, \zeta)$$
   - 多次采样 $\epsilon$ 和 $\zeta$ 形成锥形轨迹束，故名"conic"
   - 设计动机：单条 real pair 轨迹覆盖范围有限，Slerp 在高斯超球面上保持向量模长，比 Lerp 更好地保持噪声空间几何结构

3. **Slerp 噪声调度**：
   - $\zeta_{\max}$ 基于 real/fake 样本扰动重建误差的最大差异点自动确定
   - 训练过程中噪声量渐减：$\zeta(t') = \zeta^{\max} \cdot \frac{2t'^2}{1+t'^2}$
   - 周期性更新 real pair 的反演噪声以保持与最新模型的对齐
   - CIFAR-10 上 $\zeta^{\max} = 0.13$，ImageNet 上 $\zeta^{\max} = 0.23$

4. **Balanced Training 策略**：
   - 前半段训练交替进行 conic reflow（real pair）和原始 reflow（fake pair）
   - 后半段仅使用原始 reflow，补偿 fake pair 与 real pair 的数量不对称
   - 例如总步数 100 时，$U_{\text{real}} = \{1, 3, 5, \ldots, 49\}$，$U_{\text{fake}} = \{2, 4, \ldots, 50, 51, \ldots, 100\}$

### 损失函数 / 训练策略

总训练目标结合两类 pair 的 MSE 损失，其中 $\chi_{\text{fake}}$ 和 $\chi_{\text{real}}$ 为指示函数，在每个训练步中只有一个为 1。时间 $t$ 采用指数分布采样（U 形分布），重点覆盖 $t \approx 0$ 和 $t \approx 1$ 附近交叉频率高的区域。fake pair 部分使用标准 rectified flow 的速度场 MSE 损失，real pair 部分使用 conic 插值点处的速度场 MSE 损失。

## 实验关键数据

### 主实验

在 CIFAR-10 上的 one-step 和 full-step 生成质量比较：

| 方法 | NFE | IS ↑ | FID ↓ |
|------|-----|------|-------|
| 1-Rectified Flow | 1 | 1.13 | 378 |
| 2-RF Original (+Distill) | 1 | 8.08 (9.01) | 12.21 (4.85) |
| **2-RF Ours (+Distill)** | **1** | **8.79 (9.11)** | **5.98 (4.16)** |
| RF++† | 1 | 8.87 | 4.43 |
| RF++† + Ours | 1 | 8.87 | **4.22** |
| 3-RF Original (+Distill) | 1 | 8.47 (8.79) | 8.15 (5.21) |
| **3-RF Ours (+Distill)** | **1** | **8.84 (8.96)** | **5.48 (4.68)** |
| 1-RF (RK45) | 127 | 9.60 | 2.58 |
| 2-RF Original (RK45) | 110 | 9.24 | 3.36 |
| **2-RF Ours (RK45)** | **104** | **9.30** | **3.24** |

核心数据：2-rectified flow 上 one-step FID 从 12.21 降至 5.98（提升超 50%），且仅使用 300K fake pair（原始需 4M，节省 92.8%）。

在 ImageNet 64x64 和 LSUN Bedroom 256x256 上同样验证了泛化性：

| 数据集 | 方法 | Euler 1-step FID | RK45 FID |
|--------|------|-----------------|----------|
| ImageNet 64x64 | Original | 39.7 | 31.2 |
| ImageNet 64x64 | **Ours** | **37.8** | **28.2** |
| LSUN Bedroom 256 | Original | 139.98 | 24.76 |
| LSUN Bedroom 256 | **Ours** | **26.54** | **24.14** |

LSUN 上 1-step FID 从 139.98 骤降至 26.54，改善极为显著。

### 消融实验

各组件对 2-rectified flow 效果的贡献（CIFAR-10, 1-step）：

| 配置 | FID ↓ | IS ↑ | Curvature ↓ | Recon Real ↓ | Recon Fake |
|------|-------|------|-------------|-------------|------------|
| Original | 12.21 | 8.08 | 0.002837 | 0.033668 | 0.024106 |
| No Slerp (仅 real pair) | 6.60 | 8.57 | 0.002322 | 0.023380 | 0.020154 |
| Fixed Real Pair (不刷新) | 6.69 | 8.59 | 0.002313 | 0.020227 | 0.020607 |
| **Ours (完整)** | **5.98** | **8.79** | **0.002295** | **0.019404** | 0.023139 |

Slerp vs Lerp 的噪声对比：

| 方法 | IS ↑ | FID ↓ |
|------|------|-------|
| **Slerp (Ours 调度)** | **8.72** | **6.63** |
| Slerp 递增 | 8.48 | 6.64 |
| Slerp 递减 | 8.45 | 6.70 |
| Lerp 线性插值 | 8.46 | 7.50 |

### 关键发现

1. **分布漂移是 reflow 的固有问题**：在 two moons 玩具数据集上，连续 reflow 导致 KL 散度持续增长，fake 分布逐步偏离目标分布
2. **Real pair 有效纠偏**：仅加入 real pair（无 Slerp）即可将 FID 从 12.21 降至 6.60，说明真实数据锚定效果显著
3. **Slerp 优于 Lerp**：Slerp 在高斯超球面上保持模长，FID 比 Lerp 低约 0.9（6.63 vs 7.50）
4. **定期刷新 real pair 很重要**：固定不刷新反演噪声比定期更新差（6.69 vs 5.98）
5. **Recall 显著提升**：在 ImageNet 上 1-step Recall 从 0.4604 提升至 0.5325，说明方法提高了对真实分布的覆盖度
6. **极端 k=4 仍有效**：4-rectified flow 上 1-step FID 从 6.58 降至 5.66，曲率也更低

## 亮点与洞察

- **分析深刻**：不仅指出 reflow 有分布漂移问题，还通过重建误差和扰动重建误差的 real/fake 差异给出了定量证据，分析角度新颖
- **设计简洁且即插即用**：Conic reflow 不需要修改网络结构或引入判别器，仅改变训练数据的构造方式，可兼容 RF++、InstaFlow 等已有方法
- **效率极高**：对 fake pair 的需求量从 4M 降至 300K（仅需 7.2%），大幅降低 reflow 的计算成本
- **IVD 指标的提出**：除了曲率外引入 Initial Velocity Delta 评估初始速度准确性，更直接关联 1-step 生成质量
- **Slerp 噪声调度自适应确定**：$\zeta^{\max}$ 基于 real/fake 重建误差差异自动选择，避免手动调参

## 局限性 / 可改进方向

1. **仅验证了无条件生成**：未在文本条件生成（如 SD3/Flux）上验证，这些模型通常只用 1-rectified flow，方法的适用性需进一步确认
2. **ImageNet 上 real pair 数量不足**：60K real pair 对 ImageNet 的类别多样性覆盖有限，增加 real pair 可能进一步提升效果
3. **反演质量依赖原模型**：real pair 的噪声端 $v^{-1}(X_1)$ 质量取决于先导模型的反演精度，差模型可能产生低质 real pair
4. **分辨率有限**：实验最高只到 256x256（LSUN），未在更高分辨率上验证
5. **Slerp 调度的理论支撑不足**：U 形由大到小再到大的调度模式主要基于经验，缺乏理论证明其最优性

## 相关工作与启发

- **Rectified Flow / Flow Matching**：本文建立在 Liu et al. 的 rectified flow 框架上，改进了其 reflow 步骤，思路可推广到所有需要 reflow 的流匹配模型
- **Rectified++**：通过修改时间分布和损失函数改进 RF，本文方法与其正交，可直接叠加（RF++† + Ours = FID 4.22）
- **PerFlow**：用分段线性路径稳定 reflow，与本文的连续锥形路径形成对比
- **自消耗训练（Self-consuming Training）**：本文揭示的分布漂移与 self-consuming 文献中的模型坍塌有关联，conic reflow 可看作一种通过真实数据注入对抗自消耗退化的策略
- **对 idea 设计的启发**：在任何涉及"用模型自身输出做训练数据"的场景中，都应警惕分布漂移，可通过真实数据锚定来缓解

## 评分

- **新颖性**: ⭐⭐⭐⭐ - 对 reflow 分布漂移的分析视角新颖，conic reflow 的设计自然而有效
- **实验充分度**: ⭐⭐⭐⭐ - 多数据集验证 + 充分消融 + 定量漂移分析，但缺少高分辨率和条件生成实验
- **写作质量**: ⭐⭐⭐⭐ - 问题动机清晰，图示直观（尤其 Figure 1-4），公式推导完整
- **价值**: ⭐⭐⭐⭐ - 揭示了 reflow 的基本缺陷并给出简洁解决方案，对流匹配社区有实际意义；即插即用特性增加了工程价值
