# AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale

**会议**: ECCV 2024  
**arXiv**: [2404.03482](https://arxiv.org/abs/2404.03482)  
**代码**: [https://github.com/apardyl/AdaGlimpse](https://github.com/apardyl/AdaGlimpse) (有)  
**领域**: 主动视觉探索 / 高效视觉推理 / 强化学习  
**关键词**: Active Visual Exploration, Soft Actor-Critic, 可变尺度glimpse, Vision Transformer, 强化学习  

## 一句话总结
提出AdaGlimpse，利用Soft Actor-Critic强化学习从连续动作空间中选择任意位置和尺度的glimpse，结合弹性位置编码的ViT编码器实现多任务（重建/分类/分割）的主动视觉探索，以仅6%像素超越了使用18%像素的SOTA方法。

## 背景与动机
主动视觉探索（AVE）研究的是智能体如何从环境中动态选择观测区域以完成视觉任务——这对机器人、无人机等受限于有限视野和计算资源的平台至关重要。现有AVE方法存在一个关键限制：它们只能从固定网格中选择固定尺度的glimpse（要么是等大小的patch，要么是视网膜式采样），完全没有利用现代硬件的能力——PTZ相机可以自由调节焦距和视角，无人机可以改变高度来调整观测范围。这种软件能力和硬件能力之间的差距是本文要弥合的。

## 核心问题
如何让AVE智能体能够在连续空间中自由选择观测位置和缩放尺度，从而像人眼一样"先粗看全局、再细看局部"，用最少的观测次数理解场景？核心挑战在于：（1）连续的位置+尺度动作空间使采样操作不可微，无法直接梯度优化；（2）不同尺度的patch需要统一编码处理；（3）智能体需要在探索（看新区域）和利用（看重要区域）之间取得平衡。

## 方法详解
AdaGlimpse的核心思想是让智能体在连续空间中同时决定下一次观测的位置$(x,y)$和缩放尺度$z$，形成一个三元组$(x,y,z) \in [0,1]^3$。整体流程：每一步中，RL agent根据已观测信息预测下个glimpse的位置和尺度→相机捕获该区域→ViT编码器处理所有已收集的patch→得到任务预测结果和新的状态表示→循环。

### 整体框架
系统由两部分组成：（1）基于ElasticViT的ViT编码器+任务头，处理可变尺度的patch并完成下游任务；（2）Soft Actor-Critic RL agent，根据编码器输出的状态信息决定下一个glimpse的坐标和尺度。两个模块交替训练——前30个epoch只训练RL agent，之后每轮交替优化backbone和RL agent。

### 关键设计
1. **连续动作空间的自适应glimpse采样**: 与以往只能从离散网格中选取的方法不同，AdaGlimpse将glimpse坐标定义为连续值$(x,y,z)$，其中$z$控制缩放级别（$z=0$最大zoom，$z=1$最广视角）。每个glimpse以固定分辨率$d_{cam} \times d_{cam}$采样，但可覆盖不同大小的场景区域。这使得模型能够先用低分辨率看全局，再用高分辨率看细节。

2. **ElasticViT弹性位置编码**: 标准ViT假设patch来自规则网格，而AdaGlimpse的patch位置和尺度各异。采用ElasticViT的位置编码方案，根据每个patch在原始场景中的实际坐标计算位置嵌入，打破了网格限制。同时利用attention rollout估计每个patch的重要性$\hat{I}_t$，为RL agent提供信息量信号。

3. **任务自适应的解码器设计**: 分类任务直接用class token经线性层输出；密集预测（重建/分割）使用MAE风格的transformer解码器，输入所有编码器token加上全网格的mask token——与MAE不同的是这里需要预测完整图像而非仅缺失部分，因为可变尺度采样下编码器patch与解码器网格不对齐。

4. **SAC强化学习agent**: 选择Soft Actor-Critic而非其他RL算法，因为SAC的最大熵目标$V^{\pi_\theta}(s) = \mathbb{E}_\pi[\sum \gamma^t r_t + \alpha \mathcal{H}(\pi(s_t))]$天然鼓励探索多样的动作分布，非常适合需要"探索环境"的AVE任务。Actor和Critic各自包含：一个小CNN处理patch图像$\hat{G}_t$、三组MLP分别处理坐标$\hat{C}_t$、重要性$\hat{I}_t$和latent表示$\hat{H}_t$，再通过attention pooling聚合、MLP输出。两个网络不共享参数以保证训练稳定。

### 损失函数 / 训练策略
- **任务损失**: 重建用RMSE；分类和分割用教师模型的软标签+KL散度蒸馏（分类教师为DeiT-III，分割教师为DeepLabV3-ResNet101）
- **RL奖励**: 定义为相邻步骤任务损失的差值$r_t = L_{t-1} - L_t$，即每个glimpse带来的损失下降量
- **训练策略**: 先预训练600 epochs，随机采样196个glimpse让ViT学习不规则位置编码；再正式训练100 epochs，前30轮只训练RL agent，之后交替训练；使用3-Augment数据增强

## 实验关键数据

| 数据集 | 指标 | 本文(AdaGlimpse) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| ImageNet-1k (重建) | RMSE↓ | **14.5** | 30.3 (AME) | 52.1% |
| SUN360 (重建) | RMSE↓ | **11.1** | 23.6 (AME) | 53.0% |
| ADE20K (重建) | RMSE↓ | **14.0** | 23.8 (AME) | 41.2% |
| MS COCO (重建) | RMSE↓ | **14.5** | 25.2 (AME) | 42.5% |
| ImageNet-1k (分类) | Acc↑ | **77.54%** | 76.13% (STAM) | +1.41% |
| ImageNet-1k (分类+早停) | Acc↑ | 76.30% | 76.13% (STAM) | 像素少40%+ |
| ADE20K (分割, 18%像素) | PA↑ | **67.4%** | 52.4% (GlAtEx) | +15.0% |
| ADE20K (分割, ~37%像素) | IoU↑ | **25.7%** | 24.4% (AME, 56%像素) | 像素少35% |

### 消融实验要点
- **RL状态组件的重要性**（ImageNet分类）：去掉latent $\hat{H}_t$掉点最多（77.54→61.82，-15.7%），说明ViT的隐层表示是最关键的信息；去掉坐标$\hat{C}_t$也严重影响（77.54→68.25，-9.3%）；去掉重要性$\hat{I}_t$影响最小（77.54→77.36）
- **glimpse策略分析**：重建任务中模型倾向于先看全局再平均覆盖四象限；分类任务中模型在快速全局扫描后集中注意力于图像中心（ImageNet中物体多居中）
- 仅用6%像素就超越了其他方法使用18%像素的重建效果

## 亮点
- **连续动作空间是核心创新**：打破了AVE领域所有方法只能从离散网格选patch的范式，真正实现了"zoom in/zoom out"的自由度，更贴近真实硬件能力
- **SAC选型恰到好处**：SAC的最大熵特性完美匹配AVE中"探索"的需求，这个RL算法和问题的匹配非常自然
- **任务无关架构**：同一个框架无缝适配重建、分类、分割三类任务，说明"主动选择观测"的能力是任务无关的通用技能
- **先粗后细的emergent行为**：模型自然学会了"先看全局低分辨率、再zoom in看细节"的策略，与人类视觉探索模式高度一致
- 实验中展示了early stopping机制（达到阈值置信度即停止），进一步节省计算

## 局限性 / 可改进方向
- **ViT二次方复杂度**：随着glimpse数量增加，所有patch的attention计算量二次增长，作者也提到可用Mamba替代
- **静态场景假设**：所有实验基于静态图像裁切模拟，未考虑动态变化的真实环境
- **预训练开销大**：需要600 epochs随机glimpse预训练 + 100 epochs正式训练，训练成本较高
- **教师模型依赖**：分类和分割任务依赖教师模型蒸馏，限制了性能上限
- 可扩展方向：结合Mamba降低序列计算复杂度；迁移到视频或3D场景的主动探索；结合VLM实现语义引导的探索策略

## 与相关工作的对比
- **vs STAM** (CVPR 2022): STAM也用actor-critic做分类任务的glimpse选择，但只支持离散网格+固定尺度，且只用one-step AC。AdaGlimpse用连续SAC+多步探索+可变尺度，分类提升1.41%
- **vs AME** (IJCAI 2023): AME基于MAE attention map做区域选择，也限制在固定网格。重建任务上AdaGlimpse用更少像素（~24% vs 25%）实现了RMSE从30.3到14.5的巨大飞跃
- **vs AutoGaze** (CVPR 2026): AutoGaze同样受"先粗后细"启发做自回归patch选择，但面向视频MLLM的token压缩；AdaGlimpse更聚焦于真实主动探索场景的RL决策

## 启发与关联
- 与 `ideas/model_compression/20260316_adaptive_model_routing.md`（自适应模型路由）思路相通：都是"根据输入难度动态调整计算分配"，但AdaGlimpse在空间维度做自适应，model routing在模型选择维度做自适应
- 与 `ideas/object_detection/20260316_ib_adaptive_token_compression_vit.md`（IB自适应Token压缩）互补：AdaGlimpse决定"看哪里"，token压缩决定"保留哪些token"，两者可结合
- 启发新idea：将AdaGlimpse的连续尺度RL探索策略迁移到自动驾驶的多传感器注意力分配——根据场景复杂度动态决定对哪些区域用高分辨率感知

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续动作空间+SAC的组合在AVE中是首次，但整体思路（RL选区域+ViT编码）不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖重建/分类/分割三类任务、多个数据集、详细消融和可视化分析
- 写作质量: ⭐⭐⭐⭐ 思路清晰，MDP形式化完整，但部分符号定义较繁琐
- 价值: ⭐⭐⭐⭐ 对主动视觉探索领域有实质性推进，但受限于ViT复杂度和静态场景假设，实际部署仍有距离
