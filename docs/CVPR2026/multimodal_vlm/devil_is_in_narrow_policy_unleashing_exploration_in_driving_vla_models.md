# Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models

**会议**: CVPR 2026  
**arXiv**: [2603.06049](https://arxiv.org/abs/2603.06049)  
**代码**: [GitHub](https://github.com/Mashiroln/curious_vla.git)  
**领域**: 多模态VLM  
**关键词**: 自动驾驶VLA, 窄策略问题, 探索-利用困境, 强化学习, 轨迹多样性

## 一句话总结

揭示驾驶 VLA 模型中被忽视的"窄策略"（Narrow Policy）瓶颈——IL 阶段过度利用导致探索坍缩，进而限制 RL 阶段。提出 Curious-VLA 框架，通过可行轨迹扩展 + 多样性感知 RL 在 Navsim 上达到 SOTA（PDMS 90.3，Best-of-N 94.8）。

## 研究背景与动机

1. **驾驶 VLA 的两阶段范式**：当前驾驶 VLA 普遍采用 IL（模仿学习）→ RL（强化学习）的两阶段训练，但存在根本性的探索-利用失衡。
2. **窄策略问题的发现**：IL 阶段使用交叉熵损失模仿真值轨迹，导致策略分布坍缩到单一模式，多次推理产生的轨迹几乎完全重叠（mean-pFDE 仅 0.20-0.33m）。
3. **RL 阶段的优势坍缩**：策略坍缩后，GRPO 采样的奖励几乎相同（$R(y_i) \approx \mu_R$），标准差 $\sigma_R \to 0$，导致优势估计 $A_i \to 0$，梯度消失。
4. **交叉熵损失的内在缺陷**：CE 将所有非真值 token 视为等价错误，缺乏空间/功能邻近性概念，鼓励对单一模式的过度自信。
5. **时序尺度不匹配**：远期航点方差远大于近期（$t=4s$ 的方差比 $t=0.5s$ 大数量级），远期损失主导训练。
6. **行为诊断指标的缺失**：此前缺乏定量诊断窄策略现象的工具。

## 方法详解

### 整体框架

Curious-VLA 在 IL 和 RL 两个阶段均进行改进：IL 阶段通过可行轨迹扩展（FTE）增加训练数据多样性 + 步级归一化；RL 阶段通过自适应多样性感知采样（ADAS）+ 跨度驾驶奖励（SDR）维持探索。

### 关键设计

#### 可行轨迹扩展（FTE）

1. **探索性数据扩展**：从 103k NavTrain 中筛选 12k 挑战场景（多车道/路口/遮挡），利用 ReCogDrive 扰动扩散隐变量生成多样化可行轨迹，经 PDMS 评分器安全过滤，最终扩展至 142k 样本
2. **CoT 数据合成**：用 Qwen2.5-VL-72B 生成四阶段推理链（感知→解释→元行为→轨迹）
3. **步级归一化**：$\tilde{w}_t = (w_t - \mu_t) / \sigma_t$，独立归一化每个预测步以均衡各时间步的梯度幅度

#### 自适应多样性感知采样（ADAS）

将每个场景的结果建模为伯努利过程，通过 $M$ 次离线 rollout 估计成功率 $\hat{p}$。仅保留满足两个多样性条件的场景：(1) $\hat{p}^G + (1-\hat{p})^G < \epsilon_{\text{div}}$（排除全成/全败场景）；(2) $|\sigma_R - \sqrt{\hat{p}(1-\hat{p})} R_{\text{range}}| < \epsilon_{\text{conf}}$（确保奖励分布符合理论预期）。

#### 跨度驾驶奖励（SDR）

将原始 PDMS 重构为 focal-style 形式：$R_{\text{span}} = \prod_{c \in C} c \cdot \frac{\sum w'_m (1-(1-m)^{\gamma_m})}{\sum w'_m}$，通过非线性放大次优与最优行为的奖励差异。

### 损失函数

IL 阶段用标准 CE 损失（归一化后）；RL 阶段用 GRPO 目标函数 + SDR 奖励。

## 实验关键数据

### 主实验：Navsim V1 Benchmark

| 方法 | 基座 | PDMS↑ | NC↑ | EP↑ |
|------|------|-------|-----|-----|
| UniAD | - | 84.0 | 97.7 | 79.2 |
| ReCogDrive | InternVL2-8B | 89.6 | 98.2 | 83.5 |
| AutoVLA | Qwen2.5-VL-3B | 89.1 | 98.4 | 81.9 |
| AdaThinkDrive | InternVL3-8B | 90.3 | 98.4 | 84.4 |
| **Curious-VLA** | Qwen2.5-VL-3B | **90.3** | 98.4 | **88.5** |
| **Curious-VLA†(BoN)** | Qwen2.5-VL-3B | **94.8** | - | - |

### 消融实验：行为诊断

| 方法 | Diversity(pFDE)↑ | Quality(min-FDE)↓ | PDMS↑ |
|------|-------------------|-------------------|-------|
| Qwen2.5-VL | 0.20m | 1.05m | - |
| ReCogDrive | 0.33m | - | - |
| **Curious-VLA** | **最优** | **最优** | **90.3** |

### 关键发现

- BoN PDMS 94.8 直接证明了探索潜力被成功释放
- 直接对 IL 后模型进行 GRPO 训练反而降低性能，验证了窄策略对 RL 的阻碍
- 步级归一化显著提升近期航点的学习效果
- ADAS 有效避免了 RL 阶段的早期饱和

## 亮点与洞察

- **窄策略问题的发现与形式化**是重要贡献，揭示了 IL→RL 管线的根本瓶颈
- 行为诊断（Diversity/Quality/Performance）三维指标设计直观有效
- 仅用 3B 模型 + 单摄像头即达到 SOTA，效率优势明显
- BoN 评估方式巧妙验证了策略的探索潜力
- 从数据扩展、采样策略、奖励函数三个层面系统解决问题

## 局限性

- FTE 依赖 ReCogDrive 的扩散模块生成多样轨迹，引入了外部依赖
- Navsim 是闭环模拟器，真实世界效果待验证
- ADAS 的离线 rollout 阶段计算开销较大
- 核心分析基于 VLA-Token 范式，VLA-Planner 范式的窄策略程度未深入讨论

## 相关工作与启发

- 与 DeepSeek-R1/GRPO 的关系：本文揭示了 GRPO 在驾驶场景的失效原因并提出针对性改进
- 与 DAPO 的对比：DAPO 改进优势估计，Curious-VLA 从数据多样性角度解决
- "Narrow Policy" 概念可推广到其他 IL→RL 场景（如机器人操控）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
