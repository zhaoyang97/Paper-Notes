# DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving

**会议**: NeurIPS 2025  
**arXiv**: [2509.17940](https://arxiv.org/abs/2509.17940)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 端到端自动驾驶, Safety DPO, 偏好优化, 轨迹规划, NAVSIM  

## 一句话总结
提出DriveDPO两阶段框架——先通过统一策略蒸馏将人类模仿相似度与规则安全分数融合为单一监督分布，再用Safety DPO构建"看似human-like但不安全 vs 既human-like又安全"的轨迹偏好对进行策略微调——在NAVSIM上达PDMS 90.0新SOTA。

## 研究背景与动机

1. **领域现状**：端到端自动驾驶通过从原始传感器直接预测未来轨迹，避免模块化pipeline的误差累积。主流方法（VADv2、UniAD等）基于模仿学习，最小化预测轨迹与人类轨迹的几何距离。
2. **现有痛点**：（a）模仿学习无法区分"看起来像人类但不安全"的轨迹——即使与人类轨迹仅有微小偏差，也可能导致碰撞或出界；（b）MSE等对称损失函数对两个方向的偏差施加相同惩罚，但安全影响是不对称的（超越人类轨迹可能追尾，滞后则安全）；（c）近期score-based方法（Hydra-MDP等）为每条候选轨迹独立回归多个分数，但未直接优化策略分布，导致次优性能。
3. **核心矛盾**：模仿学习优化"像人"但不管"安全"；score-based方法引入安全但脱耦于策略优化。需要一种方法同时优化安全性和人类一致性，且直接在策略分布层面操作。
4. **本文要解决什么**：如何将人类轨迹模仿和规则安全信号统一到策略分布的直接优化中？
5. **切入角度**：借鉴LLM中RLHF/DPO的思想——将安全性要求转化为轨迹级偏好学习。
6. **核心idea**：统一策略蒸馏（合并模仿+安全为单一分布）+ Safety DPO（利用精心构造的偏好对做轨迹级偏好对齐）。

## 方法详解

### 整体框架
两阶段训练：**Stage 1**：统一策略蒸馏——用多视角相机+LiDAR经Transfuser编码场景，Anchor Vocabulary离散化动作空间，将模仿相似度和PDMS安全分融合为统一监督分布用KL散度训练。**Stage 2**：Safety DPO——从当前策略采样K条候选，构造安全偏好对，用DPO损失微调。

### 关键设计

1. **统一策略蒸馏（Unified Policy Distillation）**
   - 做什么：将人类模仿信号和规则安全信号合并为单一软标签分布。
   - 核心思路：对每条anchor轨迹 $a_i$ 计算模仿相似度 $\text{Sim}(a_i) = \text{Softmax}(-\|a_i - \hat{a}\|_2)$ 和PDMS安全分 $\text{PDMS}(a_i)$。用log变换放大低分差异：$p_{unified}(a_i) = \text{Softmax}(w_1 \cdot \log(\text{Sim}(a_i)) + w_2 \cdot \log(\text{PDMS}(a_i)))$。用KL散度对齐策略输出。
   - 设计动机：相比score-based方法为每条轨迹独立预测分数，统一分布在所有锚点间引入竞争机制，直接优化策略分布。log变换使低安全分轨迹的惩罚被指数放大。

2. **Safety DPO**
   - 做什么：通过轨迹级偏好学习，进一步提升策略安全性。
   - 核心思路：从策略分布采样K条候选，选得分最高的作为chosen $a_w$。rejected轨迹的选择关键——**模仿式选择**：选PDMS低于阈值 $\tau$ 但与人类轨迹最近的候选（"看起来最像人但最不安全的"）。这样的偏好对精准针对模仿学习的核心问题。用标准DPO损失微调，鼓励策略偏好安全轨迹。
   - 设计动机：朴素DPO（选最高/最低分）产生的偏好对太简单，Safety DPO的巧妙构造使模型精确学习"在看似合理的轨迹中区分安全和不安全"。

3. **Anchor Vocabulary + Transfuser架构**
   - 做什么：将连续轨迹空间离散化为N个锚点，使策略输出为离散分布。
   - 核心思路：对训练集人类轨迹做k-means聚类得到N=4096个锚点，用Fourier位置编码+MLP编码锚点，通过Cross-Attention Transformer Decoder将锚点特征与场景特征融合，最后softmax输出策略分布。
   - 设计动机：离散化使DPO框架自然适用（分布上的偏好优化），且与Anchor Vocabulary范式一致。

### 训练策略
- 感知骨干：ResNet-34 (Transfuser)
- 输入：多视角相机 + LiDAR
- Stage 1: 统一策略蒸馏预训练
- Stage 2: Safety DPO微调，迭代3轮

## 实验关键数据

### 主实验（NAVSIM Benchmark）

| 方法 | 监督类型 | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|------|---------|-----|------|-----|------|-----|-------|
| Transfuser | Human | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DiffusionDrive | Human | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| Hydra-MDP | H+Rule | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 |
| WOTE | H+Rule | 98.4 | 96.6 | 81.7 | 94.5 | 99.9 | 88.0 |
| **Ours (w/o DPO)** | H+Rule | 97.9 | 97.3 | 84.0 | 93.6 | 100.0 | 88.8 |
| **Ours (full)** | **H+Rule** | **98.5** | **98.1** | **84.3** | **94.8** | **99.9** | **90.0** |

### 消融实验

| 配置 | PDMS | 说明 |
|------|------|------|
| 纯模仿学习 | 84.0 | Transfuser baseline |
| 统一策略蒸馏 | 88.8 | +4.8 PDMS |
| + Naive DPO | 89.2 | +0.4 |
| + Safety DPO (imitation-based) | **90.0** | +1.2 |
| + Safety DPO (distance-based) | 89.8 | +1.0 |

### 关键发现
- 统一策略蒸馏单独就贡献+4.8 PDMS（84.0→88.8），证明直接优化策略分布优于独立score回归。
- Safety DPO在此基础上再提升+1.2（88.8→90.0），模仿式rejected选择优于距离式选择。
- DAC（道路合规）提升最大：从92.8→98.1（+5.3），说明安全优化对出界行为改善最明显。
- EP（前进进度）也大幅提升：79.2→84.3，说明安全约束不会牺牲任务完成效率。
- Bench2Drive闭环测试中，DriveDPO也达最高success rate 30.62%和driving score 62.02。

## 亮点与洞察
- **RLHF→自动驾驶的优雅迁移**：将DPO从LLM token级偏好适配到轨迹级偏好，概念自然且实现简洁。Safety DPO的偏好对构造（"像人但不安全" vs "既像人又安全"）精准抓住了问题核心。
- **log变换的巧妙应用**：将[0,1]的安全分映射到$(-\infty, 0]$，使低安全分的轨迹在分布中被指数级降权，比线性加权更有效地区分安全/不安全。
- **统一分布 vs 独立分数**：直接证明了在所有锚点上构建竞争性的统一分布优于为每条轨迹独立回归分数（88.8 vs Hydra-MDP的86.5）。

## 局限性 / 可改进方向
- NAVSIM是开环评测，真正的安全改善需要更多闭环验证（Bench2Drive结果有限）。
- Safety DPO的rejected轨迹选择仍需手动设计策略，可探索自动化方法。
- 依赖NAVSIM模拟器获取PDMS，可能在真实世界部署时面临sim-to-real gap。
- 仅用ResNet-34骨干，更大模型/更强感知骨干的scaling效果未验证。

## 相关工作与启发
- **vs Hydra-MDP/WOTE**: 同为rule-based监督，但它们独立回归score，DriveDPO统一为分布+DPO微调，PDMS高2.0-3.5。
- **vs DiffusionDrive**: 纯模仿学习的SOTA（88.1），DriveDPO通过加入安全信号+DPO进一步提升1.9。
- **vs TrajHF**: 类似RLHF思路但TrajHF关注驾驶风格偏好，DriveDPO聚焦安全偏好。

## 评分
- 新颖性: ⭐⭐⭐⭐ DPO迁移到自动驾驶+Safety偏好对构造是亮点
- 实验充分度: ⭐⭐⭐⭐ NAVSIM SOTA + Bench2Drive闭环 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐⭐ 为端到端自动驾驶的安全偏好对齐开辟了新方向
