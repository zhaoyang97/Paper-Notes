# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**会议**: CVPR 2026  
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)  
**代码**: [GitHub](https://github.com/MICLAB-BUPT/AWC)  
**领域**: 人体理解 / 多模态学习  
**关键词**: 3D pose estimation, modality imbalance, Shapley value, Fisher Information Matrix, multi-modal fusion  

## 一句话总结
提出基于Shapley值的模态贡献评估+Fisher信息矩阵引导的自适应权重约束(AWC)正则化方法，解决RGB/LiDAR/mmWave/WiFi四模态融合中的模态不平衡问题，在MM-Fi数据集上MPJPE比naive fusion降低2.71mm，比最佳balancing方法降低约5mm，且不引入额外可学参数。

## 背景与动机
3D人体位姿估计中，RGB受遮挡和隐私限制，非侵入式传感器(LiDAR/mmWave/WiFi)可提供补充信息。但多模态端到端训练面临模态不平衡：强模态(RGB/LiDAR)主导梯度更新，抑制弱模态(mmWave/WiFi)的优化。现有平衡方法(G-Blending/OGM-GE/AGM)专为分类设计(依赖cross-entropy)，不适用于回归任务；且常引入辅助单模态头增加模型复杂度。此外，简单加更多模态不一定提升性能——4模态融合(53.87mm MPJPE)反而比RGB+LiDAR(52.93mm)差，证实模态竞争的真实存在。

## 核心问题
如何在不引入额外参数的前提下，实现多模态回归任务中的均衡优化？需要解决两个子问题：(1) 如何在回归任务中准确评估各模态贡献(分类的cross-entropy方案不适用)；(2) 如何自适应地调节各模态学习速率以实现平衡。

## 方法详解

### 整体框架
4个模态(RGB/LiDAR/mmWave/WiFi)分别通过专用编码器(VideoPose3D/Point Transformer/MetaFi++)提取特征 → 融合模块(concatenation/MLP/attention) → 位姿回归头输出3D关节坐标。两个核心组件：Shapley模块评估模态贡献 → AWC损失在学习窗口内正则化参数更新。

### 关键设计
1. **Shapley值+Pearson相关的模态贡献评估**: 传统Shapley值用cross-entropy作利润函数，适用于分类但不适用于回归。原因：弱模态在回归中产生近常值预测(标准差极低)，用MSE/MAE评估会错误地认为其"可靠"。创新点：用Pearson相关系数代替MSE作为Shapley利润函数——衡量预测值与GT关节坐标的线性相关性而非距离，不受预测幅度影响。实验验证：RGB和LiDAR一致获得高贡献分，mmWave/WiFi分数低且随训练下降。

2. **AWC (Adaptive Weight Constraint) 正则化**: 用K-Means将4个模态按Shapley分聚类为superior(ℳ_S)和inferior(ℳ_I)两组。对每个模态编码器施加参数偏移正则：L_AWC = Σ α_m · Σ_i FIM_ii · (θ_t - θ_0)²/2。FIM对角近似衡量参数重要性——强模态早期梯度大→FIM高→正则化更强(抑制过快学习)；弱模态FIM低→正则化弱(允许继续学)。α_S > α_I确保对强模态约束更大。

3. **Learning Window**: AWC仅在前K个epoch施加(K=20最优)，之后关闭。基于观察：模态相关的关键信息在训练早期获取，后期正则化反而干扰收敛。

### 损失函数 / 训练策略
L_total = L_MPJPE + L_AWC (前K个epoch) / L_MPJPE (后续epoch)。Adam优化器，lr=1e-3，每30epoch×0.1，batch=192，共50 epochs，2×RTX 3090。

## 实验关键数据
| 方法 | Protocol 1 MPJPE↓ | PA-MPJPE↓ | Protocol 3 MPJPE↓ |
|------|-------------------|-----------|-------------------|
| MM-Fi baseline | 72.90 | 47.70 | 89.80 |
| Concatenation | 53.87 | 35.09 | 48.17 |
| G-Blending | 58.40 | 37.20 | 53.13 |
| OGM-GE | 55.51 | 35.92 | 51.68 |
| Modality-level | 53.24 | 34.81 | 53.98 |
| **Ours** | **51.16** | **34.46** | **47.55** |

### 单模态性能对比
| 模态 | MPJPE | PA-MPJPE |
|------|-------|----------|
| RGB | 63.61 | 35.75 |
| LiDAR | 66.95 | 45.70 |
| mmWave | 102.89 | 52.21 |
| WiFi | 166.92 | 97.39 |
| RGB+LiDAR | 52.93 | 34.96 |
| 全部4模态(baseline) | 53.87 | 35.09 |

### 消融实验要点
- **4模态 < RGB+LiDAR**: 证实模态竞争问题真实存在(多≠好)
- **α_S=20k, α_I=10k最优**: 强模态需更强约束，弱模态也需适量约束(防止过拟合噪声)
- **α_I=0时性能下降**: 完全不约束弱模态会导致其过拟合噪声信号
- **K=20最优**: 太短(10)正则化不足，太长(25)过度干扰后期训练
- **Shapley计算开销极低**: Concat/MLP融合策略下overhead<1%，Attention下<5.4%

## 亮点 / 我学到了什么
- **Pearson相关替代MSE做Shapley利润函数**: 在回归任务中，预测值的"幅度"和"相关性"是不同维度——弱模态可能预测幅度稳定但无信息量，Pearson相关能正确识别这种情况
- **FIM自然编码模态强弱**: 不需要额外设计模态评分机制，FIM对角项直接反映参数重要性，强模态参数的FIM值高→正则化自动更强。这是一种优雅的隐式平衡
- **"4模态不如2模态"的反直觉结果**: 定量证明了多模态融合中modality competition是真实问题，不能简单堆叠模态
- **Learning window概念**: 多模态平衡主要在训练早期关键，后期应该放开限制让模型自由收敛

## 局限性 / 可改进方向
- 仅在MM-Fi一个数据集上验证，泛化性有待检验
- 4模态中mmWave和WiFi的贡献非常有限(MPJPE>100)，是否值得融合存疑
- K-Means将4模态分为2组的方式较粗糙，更多模态时(>=8)可能需要更细粒度的分组
- Shapley值计算复杂度随模态数指数增长(2^n)，4模态还好，更多模态需近似
- 缺少与模态缺失场景(missing modality)的对比

## 与相关工作的对比
- **vs OGM-GE/AGM**: 这些方法调制dominant模态的梯度方向或大小，但忽略弱模态的过拟合风险。AWC同时约束所有模态
- **vs MMPareto**: 基于Pareto前沿优化多模态梯度，需要额外的单模态梯度计算。AWC不需要单模态头
- **vs G-Blending**: 发现G-Blending在某些设置下反而比baseline差(Table 1, Concat Protocol 1: 58.40 vs 53.87)，说明分类导向的balancing方法不适合回归

## 与我的研究方向的关联
- 多模态融合中的模态平衡问题在VLM领域同样存在(视觉vs语言的dominance)
- Shapley值+Pearson相关的评估框架可迁移到其他多模态回归任务
- FIM-guided正则化的思路与continual learning中的EWC (Elastic Weight Consolidation)有相似处

## 评分
- 新颖性: ⭐⭐⭐⭐ Shapley+Pearson解决回归中的模态评估是新颖的，AWC设计有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 3个Protocol、3种fusion策略、详细消融和超参分析，但只有MM-Fi一个数据集
- 写作质量: ⭐⭐⭐⭐ 动机清晰，观察(figure 3)有说服力，但自引较多
- 对我的价值: ⭐⭐⭐ 多模态平衡的方法论有借鉴价值，特别是FIM-guided正则化思路
