# Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation

**会议**: CVPR2026  
**arXiv**: [2603.00526](https://arxiv.org/abs/2603.00526)  
**代码**: 待确认  
**领域**: llm_alignment  
**关键词**: mesh generation, reinforcement learning, preference optimization, artist-style mesh, quadrilateral mesh, online RL

## 一句话总结
提出 Mesh-Pro，首个面向3D四边形网格生成的异步在线强化学习框架，核心算法 ARPO（Advantage-guided Ranking Preference Optimization）通过 Plackett-Luce 排名模型与优势函数加权相结合，在效率（较离线 DPO 快 3.75x）和泛化性上同时取得提升，实现 artist-style 和 dense mesh 的 SOTA 生成质量。

## 背景与动机
3D 网格生成是计算机图形学的核心任务之一。近年来，基于 autoregressive transformer 的方法（如 MeshGPT、MeshAnything 等）将网格生成建模为序列生成问题，取得了显著进展。然而，要让生成的网格达到"artist-style"——即拓扑干净、流线合理、四边形占比高——仍然是一个挑战。

强化学习（RL）已被证明能有效提升生成模型的输出质量，但在 3D mesh 生成中应用 RL 面临独特困难：

1. **离线 DPO 的局限**：现有工作（如 MeshAnything V2）使用离线 DPO 对齐 mesh 生成质量。离线 DPO 依赖预先收集的偏好数据对，但 mesh 生成的输出空间极大（顶点坐标 + 面拓扑），预先收集的偏好数据很难覆盖足够的多样性，导致泛化差
2. **训练效率低**：离线 DPO 需要先生成大量候选 mesh，人工或自动标注偏好，再训练模型——这个"生成-标注-训练"循环非常耗时
3. **mesh 特有的评估难题**：与文本/图像不同，mesh 的质量评估需要考虑几何完整性（是否有破损面）、拓扑质量（四边形比例、流线方向）等几何属性，标准 reward 设计困难
4. **资源开销**：3D mesh 模型通常参数量大（1B+），在线 RL 的采样和策略更新的计算开销显著

核心动机是：**能否设计一种高效的在线 RL 框架，能够利用实时生成的样本进行策略优化，同时避免离线 DPO 的覆盖不足问题？**

## 核心问题
如何为 3D mesh 生成模型设计高效的在线偏好优化算法，在保证收敛稳定性的同时提升泛化能力和训练效率？

## 方法详解

### 整体框架：异步在线 RL

Mesh-Pro 采用异步架构，将生成（rollout）和训练（update）解耦：

- **Rollout Workers**：多个 GPU 并行进行 mesh 采样生成，每个 worker 独立基于当前策略采样 $N$ 个候选 mesh
- **Reward Evaluator**：对生成的 mesh 计算奖励分数（基于 Ray-based reward，见下文）
- **Trainer**：异步地从 rollout buffer 中取出带奖励标注的样本进行策略更新

关键效率提升在于：rollout 和训练可以**流水线化**——trainer 在处理当前 batch 时，rollout workers 已经在生成下一轮样本。相比离线 DPO 需要"完整生成 → 标注 → 训练"的串行流程，异步架构带来 **3.75x** 的训练加速。

### ARPO：Advantage-guided Ranking Preference Optimization

ARPO 是本文的核心算法贡献。它融合了两个关键思想：

#### 1. Plackett-Luce 排名模型
给定同一输入条件生成的 $N$ 个候选 mesh $\{y_1, \ldots, y_N\}$ 及其奖励 $\{r_1, \ldots, r_N\}$，按奖励从高到低排序得到排列 $\sigma$。Plackett-Luce 模型定义了一个排名上的概率分布：

$$P(\sigma | \theta) = \prod_{i=1}^{N} \frac{\exp(\log \pi_\theta(y_{\sigma(i)}))}{\sum_{j=i}^{N} \exp(\log \pi_\theta(y_{\sigma(j)}))}$$

其中 $\pi_\theta(y)$ 表示策略模型生成 $y$ 的概率。优化目标是最大化按奖励排序的排名似然。

相比 DPO 只能处理 pairwise 偏好（一个 preferred + 一个 rejected），Plackett-Luce 模型能同时利用 $N$ 个候选的排名信息，信息利用更充分。

#### 2. 优势函数加权
为进一步提升优化效率，ARPO 引入优势函数对排名梯度进行加权。定义第 $i$ 个候选的优势值：

$$A_i = r_{\sigma(i)} - \bar{r}, \quad \bar{r} = \frac{1}{N}\sum_{j=1}^N r_j$$

ARPO 的最终损失函数为：

$$\mathcal{L}_{\text{ARPO}} = -\sum_{i=1}^{N} A_i \cdot \log P_i(\sigma | \theta) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $P_i$ 是 Plackett-Luce 概率的第 $i$ 项，$\beta$ 控制 KL 正则化强度。优势加权的直觉是：让模型将更多梯度信号分配给"明显好于均值"和"明显差于均值"的样本对，而对"差距不大"的样本减少梯度更新量，减少噪声。

### Diagonal-aware Mixed Tri-Quad Tokenization
为解决 mesh tokenization 中纯四边形表示过于严格的问题，提出混合三/四边形 tokenization：

- 四边形面通过对角线分割为两个三角形面，但共享对角线边
- 引入 diagonal-aware token 表示对角线的存在/方向，使 decoder 能在生成时区分"真正的三角形面"和"四边形面的一半"
- 这种混合表示兼顾了四边形的拓扑质量和三角形的灵活性

### Ray-based Reward
设计了基于射线检测的几何完整性奖励：

- 从多个方向发射射线穿过生成的 mesh，检测每条射线的进出交点
- 如果射线进出奇偶性不一致（如进入但未退出），标记为几何破损
- 结合四边形比例、面数、顶点分布等指标，综合计算奖励分数
- 该 reward 可自动化计算，无需人工标注

## 实验关键数据

### 定量对比

| 方法 | FID ↓ | Broken Ratio ↓ | Quad Ratio ↑ | Edge Quality ↑ | User Study ↑ |
|------|-------|-----------------|--------------|----------------|--------------|
| MeshGPT | 38.7 | 12.3% | 0% (纯三角) | 0.72 | 2.1/5 |
| MeshAnything | 31.2 | 8.1% | 68.2% | 0.78 | 3.2/5 |
| MeshAnything V2 (DPO) | 27.5 | 5.4% | 74.5% | 0.83 | 3.6/5 |
| **Mesh-Pro (ARPO)** | **23.1** | **2.1%** | **82.3%** | **0.89** | **4.3/5** |

### 效率对比

| 方法 | 训练方式 | 训练时间 | GPU 数量 | 相对加速 |
|------|---------|---------|---------|---------|
| MeshAnything V2 (offline DPO) | 离线 | ~3.75 天 | 64 | 1x |
| **Mesh-Pro (async ARPO)** | 异步在线 | **~1 天** | 64 | **3.75x** |

### 消融实验
- **ARPO vs DPO**：在同等训练步数下，ARPO 的 broken ratio 比 DPO 低 3.3%，quad ratio 高 7.8%
- **优势加权的作用**：去掉优势加权后，broken ratio 增加 1.5%，说明加权对梯度信号筛选的有效性
- **Ranking (N=4) vs Pairwise (N=2)**：使用 4 个候选排名比 pairwise 对比提升 2.1% quad ratio
- **异步 vs 同步**：异步架构在 wall-clock time 上相比同步在线 RL 快约 2x

## 亮点
- **首个 mesh 生成的在线 RL 框架**：从离线 DPO 到异步在线 RL 的范式跃迁，打开了 3D 生成领域 RL 优化的新方向
- **ARPO 算法设计精巧**：Plackett-Luce 排名模型与优势加权的结合，兼顾了信息利用效率和梯度稳定性
- **训练效率显著提升**：3.75x 加速不依赖算法技巧或近似，纯粹来自架构层面的异步流水线设计
- **几何破损率极低**：2.1% 的 broken ratio 远低于竞品，说明 ray-based reward 和 ARPO 在几何质量优化上的有效性
- **混合 tokenization**：diagonal-aware 设计是一个巧妙的工程贡献

## 局限性 / 可改进方向
1. **模型规模要求高**：1.1B 参数 + 64 GPU 的配置门槛较高，小规模场景下的适用性有待探索
2. **Reward 设计的可扩展性**：Ray-based reward 主要评估几何完整性，对更高级的审美属性（如流线方向、边环质量）的建模仍有提升空间
3. **仅限封闭 mesh**：射线奇偶性检测假设 mesh 是封闭流形，对开放 mesh（如平面、布料）不完全适用
4. **与 3D 重建流程的集成**：当前仅评估了独立生成的质量，作为 3D 重建管线后处理步骤的效果未验证
5. **异步训练的稳定性**：rollout 策略与 trainer 策略之间存在版本滞后，长时间训练时可能产生 staleness 问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 mesh 生成在线 RL 框架，ARPO 算法有新意
- 实验充分度: ⭐⭐⭐⭐ 定量/用户研究/效率对比/消融均有覆盖
- 写作质量: ⭐⭐⭐⭐ 结构清晰，技术细节完整
- 价值: ⭐⭐⭐⭐ 推动 3D 生成与 RL 对齐的交叉领域发展

