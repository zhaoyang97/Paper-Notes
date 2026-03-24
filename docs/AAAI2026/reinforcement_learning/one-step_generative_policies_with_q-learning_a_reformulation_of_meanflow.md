# One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow

**会议**: AAAI 2026  
**arXiv**: [2511.13035](https://arxiv.org/abs/2511.13035)  
**代码**: https://github.com/HiccupRL/MeanFlowQL  
**领域**: 强化学习  
**关键词**: Offline RL, MeanFlow, 生成式策略, Q-Learning, 一步采样

## 一句话总结
将MeanFlow重新形式化为残差映射 $g(a_t,b,t) = a_t - u(a_t,b,t)$，实现一步噪声→动作的生成式策略，无需蒸馏或多步ODE积分，可直接与Q-learning联合训练，在OGBench和D4RL的73个任务上取得强性能。

## 研究背景与动机
1. **领域现状**：Offline RL中需要表达强且高效的策略网络。高斯策略推理快但表达力不足（无法建模多模态动作分布），Flow/Diffusion策略表达强但需要多步推理。
2. **现有痛点**：将flow-based策略与Q-learning结合面临严重困难——多步生成需要BPTT，计算昂贵且不稳定。现有解决方案采用两阶段蒸馏（先BC训练多步策略，再蒸馏到一步），但增加复杂性且损失表达力。
3. **核心矛盾**：需要同时满足：一步推理、多模态表达、Q-learning兼容。现有方法最多满足其中两个。
4. **切入角度**：MeanFlow允许一步生成（通过平均速度场），但原始形式的"速度估计→积分"解耦设计在Q-learning中不稳定。关键是将MeanFlow重新形式化为单步残差映射。
5. **核心idea一句话**：用残差形式 $g(a_t,b,t)=a_t-u(a_t,b,t)$ 重新形式化MeanFlow，合并速度估计和动作生成为一个前向传播

## 方法详解

### 整体框架
- 给定状态 $s$ 和噪声 $e\sim\mathcal{N}(0,I)$，策略网络 $g_\theta$ 一步输出动作 $\hat{a}=g_\theta(e,b=0,t=1)$
- 用MeanFlow Identity训练 $g_\theta$（无需多步ODE）
- Q-learning通过value-guided rejection sampling优化策略

### 关键设计

1. **残差形式MeanFlow**:
   - 原始形式是 $v_{ave}=u(e,0,1)$，$\hat{a}=e-v_{ave}$（两步）——早期训练中动作常超出边界需clipping，破坏Bellman target
   - 残差形式 $g(a_t,b,t)=a_t-u(a_t,b,t)$ 合并为单步，通过zero/small-variance初始化保证早期输出在[-1,1]范围内
   - 由UAT（万能近似定理）保证 $g_\theta$ 的表达能力等价于原始 $u_\theta$

2. **MeanFlow Identity训练**: 损失 $\mathcal{L}_{MFI}(\theta) = E\|g_\theta(a_t,b,t) - \text{sg}(g_{tgt})\|_2^2$，target通过链式法则计算

3. **实用增强**: Value-guided rejection sampling + 自适应BC正则化

### 损失函数 / 训练策略
- 策略损失：MeanFlow Identity loss + BC正则化 + Q-value优化
- Critic损失：Bellman error
- 单阶段端到端训练，无需蒸馏

## 实验关键数据

### 主实验
73个任务（OGBench + D4RL），offline和offline-to-online设置下MeanFlowQL一致表现最优。

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| Naive MeanFlow (2-step) | 训练不稳定 | 早期clipping破坏Bellman |
| Naive残差形式 | 欠拟合 | 无法捕捉多模态 |
| 修正残差形式 (full) | 最优 | 稳定且表达强 |

### 关键发现
- 一步生成+单阶段训练在73个任务上表现强劲，证明不需要蒸馏
- toy实验证明修正残差形式可以捕捉多模态分布，naive形式不行

## 亮点与洞察
- **一个简单的残差重新形式化解决了flow policy + Q-learning的核心困难**：不需要BPTT、不需要蒸馏、不需要多步推理
- **早期训练稳定性的解决方案很实用**：通过zero init保证输出在有效范围内

## 局限性 / 可改进方向
- MeanFlow原本是为图像生成设计的，迁移到RL可能还有未探索的优化空间
- 仅在offline/offline-to-online验证，纯online RL未测试

## 相关工作与启发
- **vs Flow+Distill**: 两阶段训练复杂且蒸馏损失表达力。本文单阶段更简单且更强
- **vs Diffusion+BPTT**: BPTT计算贵且不稳定。本文一步推理无此问题

## 评分
- 新颖性: ⭐⭐⭐⭐ MeanFlow→RL policy的重新形式化是新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 73个任务，offline+online，非常全面
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，动机链完整
- 价值: ⭐⭐⭐⭐⭐ 解决了flow policy + Q-learning的核心困难，实用价值高
