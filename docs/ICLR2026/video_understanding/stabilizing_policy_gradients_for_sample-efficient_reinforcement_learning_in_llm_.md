# CAPO: Curvature-Aware Policy Optimization for Sample-Efficient RL in LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2510.00819](https://arxiv.org/abs/2510.00819)  
**代码**: 无  
**领域**: LLM推理 / 强化学习  
**关键词**: 策略梯度, 曲率感知, 样本效率, GRPO, 二阶优化  

## 一句话总结
CAPO 通过建模优化景观的二阶几何（仅在 LM head 最后一层计算曲率），实现 token 级别的数据筛选——拒绝会导致策略崩溃的更新，使 LLM 推理 RL 训练在激进超参数下仍保持稳定，样本效率提升 30 倍。

## 研究背景与动机
1. **现有痛点**：GRPO/PPO 用于 LLM 推理训练需要保守超参数（极低学习率、大 batch），否则策略崩溃。这浪费了大量计算资源。
2. **核心矛盾**：如何在激进训练设置下（高学习率、小 batch）保持训练稳定性？
3. **切入角度**：显式建模二阶优化几何，预测哪些 token 更新会导致不稳定，在梯度下降前过滤掉。
4. **核心 idea 一句话**：在 LM head 层用 Kronecker 近似计算 Hessian，据此筛选 token 更新方向。

## 方法详解

### 关键设计
1. **Last-Layer 曲率模型**：仅在 LM head 权重矩阵 $W \in \mathbb{R}^{K \times d_i}$ 上计算曲率，利用 top-k 采样的稀疏性将内存从 $\mathcal{O}((Kd_i)^2)$ 降至 $\mathcal{O}(\tilde{k} \cdot d_i)$。
2. **数据筛选**：对每个 token 子集计算目标偏移 $m_H(\Delta\psi)$ 和策略偏移 $m_F(\Delta\psi)$，若 $m_H$ 在安全范围且 $m_F \leq \delta_F$ 则接受更新。
3. **单调改进保证（Theorem 5.1）**：$J(\pi_{\theta+\Delta\theta}) - J(\pi_\theta) \geq \omega - C\sqrt{\delta_F}$。

## 实验关键数据

### 主实验
| 方法 | 设置 | 达到MATH~70%所需completions | 效果 |
|------|------|--------------------------|------|
| GRPO (保守) | 标准 | ~150K | 正常收敛 |
| GRPO (激进) | 激进 | 崩溃 ❌ | 策略崩溃 |
| **CAPO** | **激进** | **~5K (30×更少)** | **稳定收敛** |

### 消融实验
| 发现 | 说明 |
|------|------|
| Token 拒绝率 | 训练初期 ~8%，后降至 <2%——CAPO 干预极少 |
| 可扩展性 | Dr.CAPO, ReinCAPO 均防止崩溃 |
| 计算开销 | 可忽略 |

## 亮点与洞察
- **30× 样本效率**提升且方法极简——仅在最后一层计算曲率，token 拒绝率 <8%
- 可叠加在任何策略梯度方法上（GRPO、REINFORCE、Dr.GRPO）

## 局限性 / 可改进方向
- 仅在 Qwen2.5-Math-7B 上验证，更大模型待测试
- 阈值 $\delta_H, \delta_F$ 需要调参

## 评分
- 新颖性: ⭐⭐⭐⭐ 二阶方法在 LLM RL 中的新应用
- 实验充分度: ⭐⭐⭐⭐ 30× 提升显著，但规模有限
- 写作质量: ⭐⭐⭐⭐ 理论保证清晰
- 价值: ⭐⭐⭐⭐⭐ 大幅降低 LLM RL 训练成本
