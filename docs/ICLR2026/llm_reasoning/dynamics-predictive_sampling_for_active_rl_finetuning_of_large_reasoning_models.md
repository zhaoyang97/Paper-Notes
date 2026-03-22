# Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models

**会议**: ICLR 2026  
**arXiv**: [2603.10887](https://arxiv.org/abs/2603.10887)  
**代码**: [github.com/maoyixiu/DPS](https://github.com/maoyixiu/DPS)  
**领域**: LLM Reasoning / RL Finetuning  
**关键词**: 强化学习微调, 提示采样, 隐马尔可夫模型, 大推理模型, 在线贝叶斯推断  

## 一句话总结
将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

## 背景与动机
1. RL 微调（如 GRPO）是提升 LLM 推理能力的关键技术，但效果高度依赖训练数据质量
2. 完全求解或完全未解的 prompt 梯度信号弱（GRPO 优势函数为零），"部分求解"的 prompt 最有信息量
3. Dynamic Sampling (DS) 通过扩大候选批次 + rollout 过滤来找有效 prompt，但 rollout 开销巨大（常为微调本身的数倍）
4. History Resampling (HR) 仅过滤完全求解的 prompt，粒度太粗
5. **核心问题**：如何在不进行昂贵 rollout 的前提下，在线预测哪些 prompt 当前处于"部分求解"状态？

## 方法详解

### 动力系统建模
- 每个 prompt 定义三个隐状态：完全未解(1)、部分求解(2)、完全求解(3)
- 状态转移建模为隐马尔可夫模型 (HMM)，转移矩阵 $\Phi$ 为随机变量

### 在线贝叶斯推断（三步流水线）
1. **观测更新**：若 prompt 被选中 rollout，用 Bayes 规则将先验 $\mu_t^{\text{prior}}$ 更新为后验 $\mu_t^{\text{post}}$
2. **转移更新**：用 Dirichlet-Categorical 共轭更新转移矩阵后验，引入指数衰减 $\lambda$ 处理非平稳性
3. **下一步预测**：$\mu_{t+1}^{\text{prior}} = \Phi_t \mu_t^{\text{post}}$，预测下一步求解状态

### Prompt 采样
- 按 $\mu_t^{\tau,\text{prior}}(2)$（预测为部分求解的概率）排序，选 Top-B 个 prompt 构成训练批次
- 非平稳衰减隐式提供探索：长期未采样的 prompt 预测漂移至均匀分布，自然被重新访问

## 实验

### 数学推理（MATH → AIME24 等）
| 方法 | Avg↑ | Rollouts↓ | Runtime↓ |
|------|------|-----------|----------|
| R1-1.5B+US | 48.57 | 737k | 27h |
| R1-1.5B+DS | 52.00 | **2933k** | 89h |
| R1-1.5B+**DPS** | **52.13** | 737k | 32h |
| R1-7B+US | 59.31 | 287k | 30h |
| R1-7B+DS | 62.42 | **1147k** | 73h |
| R1-7B+**DPS** | **63.13** | 287k | 39h |

### Countdown 规划 & Geometry 视觉几何
- DPS 在全部任务上匹配或超越 DS oracle，rollout 量仅为 DS 的 ~25%
- 有效样本比例（部分求解占比）DPS 达 ~90%，远高于 US 和 HR

### 预测精度
- 整体预测准确率高，Class 2 的 precision/recall/F1 均稳健
- 混淆矩阵随训练推进对角线增强

### 消融
- 非平稳衰减 $\lambda$：去除（$\lambda=1$）或极端（$\lambda=0$）均降低性能，中等值最优
- 状态划分：3 状态最优，过粗或过细均下降
- 响应组大小 $k$：$k=4$ 时 DPS 优势最大（US 有效样本比例极低）

## 亮点
- 将 prompt 采样问题优雅地转化为 HMM 状态预测问题，理论清晰
- 推断开销极低（低维矩阵运算），不增加实质计算负担
- 以 DS ~25% 的 rollout 量达到同等或更优性能，实用价值突出
- 非平稳衰减机制隐式提供探索，避免采样死锁

## 局限性
- 依赖正确性二值奖励定义状态，未验证密集奖励/过程奖励场景
- Top-k 贪心选择策略可能次优，可探索熵优先等策略
- HMM 转移矩阵为每个 prompt 独立维护，prompt 间共享结构未利用

## 相关工作
- **Dynamic Sampling (DS)**：rollout 过滤 oracle，高效但计算代价大，本文替代目标
- **History Resampling (HR)**：epoch 级吸收态假设过于刚性
- **GRPO**：本文采用的 RL 微调算法基础
- **课程学习**：DPS 与课程学习有隐式联系（按难度动态调整采样）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
