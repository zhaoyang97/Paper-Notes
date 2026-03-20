# Understanding and Improving Hyperbolic Deep Reinforcement Learning

**会议**: ICLR2026  
**arXiv**: [2512.14202](https://arxiv.org/abs/2512.14202)  
**代码**: [GitHub](https://github.com/Probabilistic-and-Interactive-ML/hyper-rl)  
**领域**: llm_agent / reinforcement learning  
**关键词**: 双曲空间, PPO, 梯度分析, RMSNorm, 分类值损失  

## 一句话总结
通过形式化梯度分析揭示双曲深度 RL 的训练不稳定根源（大范数嵌入导致信赖域违反），提出 Hyper++ 三组件方案（RMSNorm + 学习缩放 + 分类值损失）实现稳定训练并超越现有方法。

## 背景与动机
1. 序贯决策本质上产生层级数据：每个状态分支为指数多的后续状态，形成树状结构
2. 欧几里得空间体积多项式增长，无法无损嵌入指数增长的层级结构——几何失配
3. 双曲空间体积指数增长，天然适合层级嵌入，在分类/度量学习中已有成功应用
4. 但双曲深度 RL 面临严重优化困难：非平稳性放大梯度不稳定，缺乏形式化分析
5. 在 PPO 中，Poincaré Ball 的保角因子 λ 随嵌入靠近边界而爆炸，导致信赖域违反
6. 现有缓解方案（SpectralNorm + S-RYM 缩放）限制整个编码器的表达能力

## 方法详解
**Hyper++ 三大组件**：

1. **RMSNorm 正则化**（替代 SpectralNorm）
   - 仅在编码器最后线性层的预激活输出应用 RMSNorm + 1/√d 缩放
   - 保证嵌入范数有界（Proposition 4.2）：‖x̂‖₂ < 1（对 ReLU/TanH）
   - 不限制编码器其他层的表达能力（vs SpectralNorm 需全层应用）

2. **可学习特征缩放**
   - 学习标量 ξ_θ 通过 sigmoid 缩放正则化后的嵌入
   - 将 Poincaré Ball 可用半径从 0.76 扩展到 0.95，体积增益 (0.95/0.76)^d
   - d=32 时增加约 1200 倍可用体积

3. **分类值损失**（HL-Gauss 替代 MSE）
   - 双曲 MLR 层输出超平面距离，适合分类而非回归
   - 与双曲几何的超平面距离天然对齐
   - 稳定非平稳目标下的 critic 学习

**模型选择**：使用 Hyperboloid 模型替代 Poincaré Ball，避免保角因子不稳定

## 实验关键数据
**ProcGen (PPO, 16 环境)**：
- Hyper++ IQM: 0.40 vs Hyper+S-RYM: 0.26 vs Euclidean: 0.30（测试集）
- 训练回报提升 52%，前向传播时间减少 ~30%
- 消融：去掉 RMSNorm → 完全学习失败；去掉缩放 → 明显下降

**Atari-5 (DDQN)**：
- Hyper++ 在全部 5 个游戏的所有聚合指标上显著超越欧几里得和双曲基线
- NameThisGame 和 Q*bert 增益最大

**关键消融发现**：
- SpectralNorm（全层或仅倒数层）→ 均无法学习
- 欧几里得 + HL-Gauss → 反而不如 MSE（分类损失专为双曲设计）
- 欧几里得 + 全套正则化 IQM=0.35 < Hyper++ (Hyperboloid) IQM=0.40

## 亮点
- **首次系统梯度分析**：闭式推导 Poincaré Ball 和 Hyperboloid MLR + 指数映射的梯度
- 证明 SpectralNorm 不足（Lemma 4.1：需全层应用才有效，但限制表达力）
- Proposition 4.2 保证 RMSNorm 的范数有界性质，理论驱动的设计
- 组件间协同效应：双曲几何 + HL-Gauss 比各自单独更优

## 局限性 / 可改进方向
- 聚焦优化视角，未分析双曲表示实际学到了什么层级结构
- 未研究哪些环境最适合双曲表示
- 几何选择与不同 RL 算法的交互未探索
- ProcGen Phoenix 上出现可塑性丧失现象，未深入讨论

## 与相关工作的对比
- vs Cetin et al. (2023) Hyper+S-RYM：消除 SpectralNorm 的表达力-稳定性权衡
- vs Euclidean PPO/PPG/DDQN：双曲表示 + 正确正则化 = 一致性优势
- vs 双曲深度学习 (Ganea, Shimizu, Bdeir)：首次系统解决 RL 中的双曲优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度分析驱动的双曲 RL 修复，理论+实践结合
- 实验充分度: ⭐⭐⭐⭐⭐ ProcGen 16 环境 + Atari-5 + 大量消融 + 多 RL 算法
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，图表质量高
- 价值: ⭐⭐⭐⭐ 为双曲深度 RL 提供可靠的实践方案
