# HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification

**会议**: CVPR 2025 (Workshop)  
**arXiv**: [2603.12693](https://arxiv.org/abs/2603.12693)  
**代码**: [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) / [VD](https://github.com/kdtsypliakova/ABAW-10-Violence-Detection)  
**领域**: 人脸情感识别 / 情感计算  
**关键词**: 面部表情识别, Valence-Arousal估计, Action Unit检测, 暴力检测, EfficientNet, MLP

## 一句话总结
HSEmotion 团队在 ABAW-10 竞赛中提出了一个轻量级 pipeline：用预训练 EfficientNet 提取面部 embedding，结合 MLP + GLA（Generalized Logit Adjustment）+ 滑窗平滑，在四项任务（EXPR/VA/AU/VD）上均大幅超过官方 baseline，其中暴力检测任务使用 ConvNeXt-T + TCN 达到 0.783 macro F1。

## 研究背景与动机
1. **领域现状**：ABAW（Affective Behavior Analysis in-the-Wild）竞赛是情感计算的主流 benchmark，第10届包含四项任务：面部表情分类（EXPR）、Valence-Arousal 连续估计（VA）、Action Unit 检测（AU）、细粒度暴力检测（VD）。
2. **核心挑战**：真实场景数据面临遮挡、姿态/光照变化、域偏移、标签噪声、类别不平衡等问题。现有方法通常需要复杂的时序建模（Transformer/TCN）和多模态融合，计算成本高。
3. **本文切入角度**：不追求架构复杂度，而是用高质量预训练 encoder + 简单 MLP + 后处理技巧（GLA、置信度过滤、滑窗平滑）实现 "简单但有效" 的 pipeline。
4. **核心思路**：预训练模型已具有很强特征提取能力，关键在于如何高效利用这些特征并处理类别不平衡和帧间噪声问题。

## 方法详解

### 面部表情分类（EXPR）
1. **特征提取**：用 EmotiEffNet-B0（EfficientNet 在 AffectNet 上预训练）提取面部 embedding
2. **MLP 分类器**：单隐层 MLP，训练用加权 softmax loss 处理类别不平衡
3. **GLA（Generalized Logit Adjustment）**：在验证集上搜索每类 bias $b_y^*$，最大化 F1 score，有效校正类先验偏差
4. **置信度过滤**：若预训练模型输出的最高概率 $> p_0$（0.8-0.9），直接采用其预测；否则用 MLP 分类
5. **时序平滑**：滑窗对相邻帧概率取平均，消除帧级噪声
6. **可选音频融合**：提取 wav2vec 2.0 特征，训练单独 MLP，与视觉分支加权融合

### Valence-Arousal 估计（VA）
- 用 MT-DDAMFN 预训练模型提取 embedding，无隐层 MLP 回归
- 损失函数结合 MSE 和 CCC（Concordance Correlation Coefficient）
- 同样使用滑窗平滑

### Action Unit 检测（AU）
- 12 个 AU 的多标签分类，MLP + sigmoid 输出
- 使用加权 BCEWithLogitsLoss，正类权重按频率计算
- 创新点：融合 embedding 和 logits 两个 MLP 的预测（blending）
- 逐 AU 搜索最优阈值（而非统一 0.5）

### 暴力检测（VD）
- 最佳单流模型：ConvNeXt-T（ImageNet-1K 预训练）提取 768-d 帧特征 + 5 层 dilated TCN
- 多模态变体：加入 MediaPipe Pose 骨架特征（406-d → 256-d），cross-attention 融合 + BiLSTM
- 训练使用 AdamW + OneCycleLR + TrivialAugmentWide，正类权重 1.15

## 实验关键数据

### EXPR 分类（AffWild2 验证集）
| 方法 | F1-score | Accuracy |
|------|----------|----------|
| Baseline VGGFACE | 25.0 | - |
| EmotiEffNet, GLA, 滑窗 | 44.85 | 55.41 |
| EmotiEffNet, GLA, 过滤+滑窗 | 45.79 | 55.69 |
| EmotiEffNet + wav2vec, GLA, 过滤+滑窗 | **47.40** | **57.98** |
| 对比：CLIP+TCN [68] | 46.51 | - |

### VA 估计（AffWild2 验证集）
| 方法 | CCC_V | CCC_A | $P_{VA}$ |
|------|-------|-------|----------|
| Baseline ResNet-50 | 0.24 | 0.20 | 0.22 |
| MT-DDAMFN, MLP, 滑窗 | 0.510 | 0.615 | 0.562 |
| 对比：CLIP+TCN [68] | 0.562 | 0.612 | 0.587 |

### AU 检测（AffWild2 验证集）
| 方法 | F1-score |
|------|----------|
| Baseline VGGFACE | 39.0 |
| EmotiEffNet, logits+embeddings, 滑窗, 最优阈值 | **54.7** |
| 对比：CLIP+TCN [68] | 58.0 |

### 暴力检测（DVD 验证集）
| 方法 | F1_V | F1_NV | Macro F1 |
|------|------|-------|----------|
| Baseline ResNet-50 + BiLSTM | 0.56 | 0.71 | 0.640 |
| ConvNeXt-T + TCN | 0.738 | 0.828 | **0.783** |
| ConvNeXt-T + Skel. attn + BiLSTM | 0.715 | 0.828 | 0.772 |

### 关键发现
- 2D 预训练 encoder + 简单时序头一致优于 3D video backbone（如 SlowFast、VideoMAE）
- 光流双流融合反而不如纯 RGB ConvNeXt-T
- GLA 对类别不平衡的校正效果显著（F1 从 38.68 → 41.40）
- 置信度过滤 + 滑窗平滑分别贡献约 1-2% 的 F1 提升
- 逐 AU 阈值搜索比统一 0.5 阈值持续提升约 0.2-0.5%

## 亮点
- **极致的工程简洁性**：整个 EXPR pipeline 只有预训练 encoder + 单层 MLP + 三个后处理技巧，却达到接近 SOTA 的性能
- **GLA 的有效应用**：将 post-hoc logit 调整从通用分类迁移到情感识别场景，简单有效
- **置信度过滤的直觉**：预训练模型已经对高置信样本有准确判断，低置信样本才需要额外分类器
- **VD 任务的系统性消融**：测试了大量 backbone/temporal head/多模态组合，结论清晰

## 局限性
- 方法论创新有限——主要是已有技术（EfficientNet + MLP + GLA + 平滑）的工程组合
- EXPR/VA/AU 三项任务与历年 ABAW 最优方案仍有差距（尤其是 AU 检测落后 CLIP+TCN 约 3.3%）
- VA 估计仅用单帧 + 简单平滑，未充分利用时序依赖
- 音频模态集成较粗糙（简单加权融合），未探索 cross-attention 等更精细的融合方式
- 暴力检测仅在 DVD 数据集上测试，泛化性待验证

## 评分
- 新颖性: ⭐⭐ 方法论创新有限，主要是成熟技术的组合
- 实验充分度: ⭐⭐⭐⭐ 四项任务均有详细消融，VD 对比了大量架构
- 写作质量: ⭐⭐⭐ 结构清晰但偏 technical report 风格
- 价值: ⭐⭐⭐ 作为竞赛方案有工程参考价值，展示了"简单方法"的上限
