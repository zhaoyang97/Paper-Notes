# Sparse Imagination for Efficient Visual World Model Planning

**会议**: ICLR 2026  
**arXiv**: [2506.01392](https://arxiv.org/abs/2506.01392)  
**代码**: 无（基于 DINO-WM 框架）  
**领域**: Robotics / World Model / Planning  
**关键词**: world model, sparse tokens, MPC, DINO, VLA, token dropout, planning efficiency  

## 一句话总结
提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中随机丢弃 token 以大幅加速推理（50% 丢弃率减少约 50% 时间），同时通过随机分组注意力训练保持任务性能不变。

## 背景与动机
1. 基于世界模型的规划（MPC）通过想象未来轨迹实现决策，但计算代价随 token 数量二次增长
2. ViT patch token 作为视觉状态表示（如 DINO-WM）比单一 CLS token 保留更丰富的空间信息
3. 但全量 patch token 在 MPC 中需要 K×M×H 次前向传播（候选数×迭代数×规划步长），实时部署极其困难
4. ViT 表示存在已知的冗余性——并非所有 patch 对规划都同等重要
5. 机器人场景下计算资源尤其受限，需要在保持精度的同时降低推理开销
6. 现有 token 剪枝方法（注意力/学习排序/聚类合并）在规划中存在"盲点问题"

## 方法（框架/设计）
- **Sparse Imagination**: 在世界模型推理阶段随机丢弃比例为 $p$ 的 patch token，仅用 $(1-p)N$ 个 token 进行前向预测
- **随机分组注意力训练**: 训练时将每帧 token 随机分为两组，注意力掩码限制组内交互，使模型学会处理任意 token 子集
- **MPC 集成**: 每个规划步重新采样 dropout mask，预测和 CEM 优化均在稀疏 token 上进行
- **损失函数**: 标准 MSE 预测损失 $\mathcal{L}_{wm} = \frac{1}{N}\sum\|{\hat{z} - z}\|^2$，目标距离同样用 MSE
- **VLA 引导规划**: 对长时程任务从预训练 VLA（SmolVLA）采样候选动作序列，替代随机采样
- **关键发现**: 简单随机采样优于复杂的注意力/学习排序方法，因为静态重要性度量在动态规划中存在"盲点"，随机采样的无偏覆盖反而更鲁棒

## 实验关键数据
| 设置 | 性能 | 时间节省 |
|------|------|----------|
| PushT (50% drop) | 70.0% vs Full 75.0% | 82s vs 173s (−52.6%) |
| Granular (30% drop) | 85.0% vs Full 75.0% | 性能反超 |
| 真实 PickPlace (50% drop) | 80% vs VLA-only 60% | 10.4s vs 19.1s |
| 真实 Drawer (50% drop) | 70% vs VLA-only 60% | 10.6s vs 14.0s |
| LIBERO-10 (50% drop) | 33% vs VLA-only 29% | 29.7s vs 53.4s |
| Meta-World (50% drop) | 47.73% vs Full 48.80% | 2.37s vs 3.63s |

- CLS-token 基线在空间敏感任务上严重退化（Granular 20%, Rope 36.7%）
- 10%-50% 丢弃率为最佳工作区间，>70% 开始明显退化
- 分组注意力训练是关键（消融实验确认）

## 亮点
- 极其简洁优雅：仅通过随机 dropout 即实现大幅加速，无需额外模型
- "盲点问题"分析深刻——解释了为何复杂 token 选择不如随机采样
- 通用性强：从简单轨迹优化到 VLA 引导规划到真实机器人均验证有效
- 训练阶段的分组注意力策略可无缝嵌入任何 Transformer 世界模型

## 局限性
- 最佳 drop ratio 需要根据任务手动选择，缺乏自适应机制
- 分组数固定为 2，未探索更多分组的效果
- 依赖 DINO 特征的冗余性假设，对信息密集场景可能不成立
- 真实世界验证仅限于两个简单任务（PickPlace + Drawer）

## 相关工作
- **世界模型**: Dreamer 系列 (Hafner et al.)、DINO-WM (Zhou et al. 2024)
- **视觉 token 效率**: ToMe (Bolya et al.)、LTRP (Luo et al.)、DynamicViT (Rao et al.)
- **VLA**: SmolVLA (Shukor et al. 2025)
- **视觉表示**: DINO (Caron et al.)、R3M (Nair et al.)

## 评分
- 新颖性: ⭐⭐⭐⭐ (简单但有效的洞察，盲点问题分析有价值)
- 实验充分度: ⭐⭐⭐⭐⭐ (8个仿真+2个真实任务，多方法对比，消融充分)
- 写作质量: ⭐⭐⭐⭐ (逻辑清晰，图表精美)
- 价值: ⭐⭐⭐⭐ (实用贡献，可直接集成到现有世界模型流水线)
