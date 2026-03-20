# Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation

## 基本信息
- **arXiv**: 2510.21003
- **会议**: NeurIPS 2025
- **作者**: Enshu Liu, Qian Chen, Xuefei Ning, Shengen Yan, Guohao Dai, Zinan Lin, Yu Wang
- **机构**: Tsinghua University, Microsoft Research
- **代码**: https://github.com/imagination-research/Distilled-Decoding-2

## 一句话总结
提出 Distilled Decoding 2 (DD2)，通过条件分数蒸馏损失将图像自回归模型压缩为单步生成器，在 ImageNet-256 上 FID 仅从 3.40 增至 5.43，比 DD1 的 one-step 差距缩小 67%，训练加速 12.3×。

## 背景与动机
图像 AR 模型（如 LlamaGen, MAR, VAR）在质量上接近扩散模型，但需要大量采样步骤（数百到数千个 token 逐个生成）。Distilled Decoding 1 (DD1) 首次尝试将 AR 模型压缩为少步生成，但存在两个问题：
1. One-step 设置下性能退化严重
2. 依赖预定义的高斯映射，限制了灵活性

## 核心问题
如何将图像 AR 模型蒸馏为真正的单步生成器，同时保持高质量？

## 方法详解

### 1. 核心思想：条件分数蒸馏
将原始 AR 模型视为**教师模型**，其在每个 token 位置提供条件分数（conditional score）：
- 在 latent embedding 空间中，AR 模型在位置 $t$ 给出条件分布 $p_\text{teacher}(z_t | z_{<t})$
- 该条件分布的梯度就是条件分数

### 2. Conditional Score Distillation Loss
训练一个独立的 **one-step generator** 网络：
- 学生一次性并行预测所有 token
- 对每个 token 位置，利用教师 AR 模型的条件分数做蒸馏
- 在每个 token 位置 $t$，以前面已生成的 token $z_{<t}$ 为条件，计算学生输出与教师条件分布的对齐损失
- 不需要预定义映射（vs. DD1 的高斯映射约束）

### 3. 分数预测网络
- 训练独立网络预测生成分布的条件分数
- 条件分数引导学生沿正确方向更新
- 本质上是在 token 空间中做"score matching + distillation"

### 4. vs. DD1 的关键改进
| 对比 | DD1 | DD2 |
|---|---|---|
| 映射方式 | 预定义高斯映射 | 无需预定义映射 |
| 蒸馏方式 | 确定性映射蒸馏 | 条件分数蒸馏 |
| One-step 质量 | 退化严重 | FID 仅增 2.03 |
| 训练效率 | baseline | **12.3× 加速** |

## 实验关键数据

### ImageNet-256 生成质量
| 方法 | Steps | FID↓ |
|---|---|---|
| 原始 AR 模型 | 256 | 3.40 |
| DD1 (one-step) | 1 | ~9.5 (估计) |
| **DD2 (one-step)** | **1** | **5.43** |

- 比 DD1 的 one-step 差距缩小 **67%**
- 训练速度提升 **12.3×**

### 关键优势
- 单步生成速度比原 AR 模型快 ~256×
- FID 仅增加 2.03（3.40 → 5.43），保持高质量

## 亮点
1. **单步 AR 生成的突破**：首次让图像 AR 模型真正实现高质量单步生成
2. **条件分数蒸馏**：优雅地将 AR 教师的逐 token 知识提炼到并行学生
3. **无需预定义映射**：比 DD1 更灵活通用
4. **12.3× 训练加速**：兼顾效率和质量
5. **与 InfinityStar 互补**：InfinityStar 用 next-scale 加速 AR，DD2 用蒸馏进一步压缩到单步

## 局限性
1. 仍需要原始 AR 模型作为教师（训练时开销）
2. 单步生成的 FID (5.43) 仍不及多步 AR (3.40)
3. 主要在 ImageNet-256 上验证，T2I 场景未涉及
4. 条件分数预测网络的设计可能影响泛化

## 与相关工作的对比
- **vs. DD1**：DD2 去掉了预定义映射限制，one-step 质量差距缩小 67%
- **vs. Consistency Models**：后者针对扩散模型的 one-step 蒸馏，DD2 针对 AR 模型
- **vs. InfinityStar**：InfinityStar 用 next-scale prediction 减少 AR 步数到 K，DD2 进一步压缩到 1
- **vs. SANA-Sprint**：SANA-Sprint 是 T2I 扩散的步数压缩，DD2 是 AR 的步数压缩

## 启发与关联
- **AR 模型加速的终极形态**：从数百步 → next-scale (~20步) → one-step，DD2 代表了 AR 加速的极限探索
- **Score distillation 的通用性**：将 SDS (Score Distillation Sampling) 从 3D 生成移植到 AR 蒸馏，展示了跨领域迁移
- **与 Does Thinking More Help? 的联系**：推理模型也可能受益于类似的"多路蒸馏"——与其延长单条推理链，不如蒸馏多步的知识到更短的过程

## 评分
- 新颖性：★★★★☆ — 条件分数蒸馏应用于 AR 模型是新贡献
- 技术深度：★★★★☆ — 分数蒸馏+条件化设计严谨
- 实验完整度：★★★★☆ — ImageNet-256 验证充分但场景单一
- 写作质量：★★★★☆ — 与 DD1 的对比清晰
