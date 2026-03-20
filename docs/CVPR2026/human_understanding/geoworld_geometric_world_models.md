# GeoWorld: Geometric World Models

**会议**: CVPR 2026  
**arXiv**: [2602.23058](https://arxiv.org/abs/2602.23058)  
**代码**: [https://steve-zeyu-zhang.github.io/GeoWorld](https://steve-zeyu-zhang.github.io/GeoWorld)  
**领域**: 世界模型 / 视觉规划 / 双曲空间  
**关键词**: 世界模型, 双曲空间, JEPA, 几何强化学习, 多步规划  

## 一句话总结
在V-JEPA 2中引入双曲流形表示（Hyperbolic JEPA）和几何强化学习（GRL），利用测地线距离编码层次关系，通过能量函数优化实现更稳定的长时域规划，3步规划提升约3% SR，超越GPT-5 zero-shot。

## 背景与动机
预测式世界模型（JEPA）在欧式空间学表示，忽略状态间层次结构。多步规划中未来状态指数增长（$B^d$）天然形成树——双曲空间适合编码这种层次。单步训练的预测器在长时域rollout中误差快速累积。

## 核心问题
(1) 欧式空间无法自然编码层次关系；(2) 需要RL机制优化长时域规划。

## 方法详解
### 整体框架
视频帧 → 冻结V-JEPA 2 encoder → 指数映射$\exp_0$到双曲空间 → 动作条件预测器（300M Transformer）→ CEM能量最小化规划

### 关键设计
1. **Hyperbolic JEPA**: Poincaré球模型，$\exp_0(v)=\tanh(\sqrt{c}\|v\|)\frac{v}{\sqrt{c}\|v\|}$，曲率$c$可学习（收敛~0.3）
2. **GRL**: 路径价值函数$V^*=\min_\phi\sum\gamma^{t-1}d_\mathbb{H}(\hat{s}_{t+1},s_{t+1})$ + 三角不等式正则$\mathcal{L}_\Delta$
3. **双阶段训练**: 监督SFT($\lambda=0.5$平衡单步+rollout loss)→GRL(能量优化)

### 损失函数 / 训练策略
SFT: $\lambda\mathcal{L}_{TF}+(1-\lambda)\mathcal{L}_{rollout}$，双曲测地线距离；GRL: $\sum\gamma^{t-1}d_\mathbb{H}+\beta\mathcal{L}_\Delta$；4节点×8 H100

## 实验关键数据
| 方法 | CrossTask SR(T=3/T=4) | COIN SR(T=3/T=4) |
|------|----------------------|-------------------|
| V-JEPA 2 ViT-g384 (video) | 50.16/35.01 | 42.74/31.63 |
| GeoWorld ViT-g384 | **51.71/37.04** | **45.29/33.29** |
| GPT-5 (zero-shot) | 50.03/30.20 | 43.84/32.64 |

### 消融实验要点
- SFT+GRL互补（50.42→51.04→51.71）；曲率收敛~0.3
- T=3→T=8：GeoWorld降速明显慢于V-JEPA 2
- Gromov δ-hyperbolicity：GeoWorld更集中于零→更强树状结构

## 亮点 / 我学到了什么
- 双曲空间天然适合多步规划层次建模
- 三角不等式正则是优雅几何约束
- 超越GPT-5 zero-shot证明了领域特定世界模型的价值
- 能量景观可视化直观展示欧式vs双曲差异

## 局限性 / 可改进方向
- 层次来自隐式展开而非显式子任务；未验证embodied场景
- → 可与 `ideas/20260316_concept_bottleneck_world_model.md` 和 `ideas/20260316_streaming_world_scene_graph.md` 关联

## 与相关工作的对比
vs V-JEPA 2: 同架构双曲+GRL持续提升；vs VideoWorld: 生成式长时域不如预测式；vs GPT-5: 纯视觉世界模型超越超大VLM

## 与我的研究方向的关联
双曲表示学习在视觉层次建模中有前景；几何RL思路可用于视觉模型后训练

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双曲世界模型+几何RL全新范式
- 实验充分度: ⭐⭐⭐⭐ 双数据集多尺度完整消融
- 写作质量: ⭐⭐⭐⭐ 概念清晰，补充材料含完整教学
- 对我的价值: ⭐⭐⭐⭐ 双曲几何和能量景观对理解表示结构有深层启发
