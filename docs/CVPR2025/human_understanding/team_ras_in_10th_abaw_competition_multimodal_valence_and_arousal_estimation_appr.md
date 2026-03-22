# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**会议**: CVPR2025  
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)  
**代码**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)  
**领域**: human_understanding  
**关键词**: valence-arousal estimation, multimodal fusion, mixture of experts, affective computing, ABAW

## 一句话总结

提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

## 研究背景与动机

1. **连续情感估计挑战**：在野外条件下进行 valence-arousal 连续估计面临外观变化、头部姿态、光照、遮挡和个体差异等困难
2. **视觉模态主导**：现有 ABAW 竞赛方法主要依赖视觉特征（ResNet、ViT、EfficientNet），但单模态信息不够充分
3. **VLM 潜力未开发**：多模态视觉语言模型（VLM）在情感分析中的应用仍待探索，尽管其能捕获上下文和情境情感线索
4. **音频模态的不可靠性**：Aff-Wild2 面向视频收集，音频通常嘈杂不可靠，包含大量非语音片段
5. **模态融合复杂性**：不同模态在不同时刻的可靠性不同，需要自适应加权而非简单拼接
6. **行为描述新维度**：利用 VLM 提取面部表情、头部运动、手势、姿势等行为级别语义信息作为情感估计的补充

## 方法详解

### 整体框架

三模态管线：Face（帧级特征 + 时序回归）、Behavior（VLM 段级嵌入 + 时序建模）、Audio（语音特征 + 分块池化），经两种融合策略之一输出 valence/arousal 预测。

### 面部模型

- **特征提取**：YOLO 人脸检测 → GRADA 编码器（EfficientNet-B1 基础，在 10 个情感数据集上多任务微调，输出 256 维嵌入）
- **时序建模**：Transformer-based 序列回归模型，投影层 → 多层 Transformer → 回归头（FCL + LN + GELU + Dropout + FCL）
- **滑窗处理**：重叠时序窗口（L=400, S=150），未检测到人脸时使用最近帧填充

### 行为描述模型

- **特征编码**：Qwen3-VL-4B-Instruct 处理每段 16 帧均匀采样 + 情感导向文本 prompt，提取最后隐藏层 last token 表示
- **两种设置**：visual-only 嵌入 vs multimodal（视频+文本）嵌入，后者明显更优
- **时序建模**：段级嵌入经 Mamba 块堆叠建模时序动态（4-12 层, hidden 128-256, state size 8）
- **帧级展开**：段级预测扩展到对应帧区间，重叠段取平均

### 音频模型

- **预处理**：4 秒分段 + 2 秒重叠，mono 16kHz
- **跨模态过滤**：基于 MediaPipe 嘴部开合检测过滤非语音片段（时序平滑 + 开口时长阈值）
- **特征提取**：WavLM-Large（MSP-Podcast 预训练），仅微调顶部 4 层 Transformer
- **分块池化**：隐藏表示分 4 个时间块，每块用 attention-statistics pooling 聚合（加权均值 + 加权标准差）

### 融合策略

**DCMMOE（Directed Cross-Modal Mixture-of-Experts）**：
- 所有有序模态对构成交叉注意力专家 $|\mathcal{E}| = M(M-1)$，显式建模非对称跨模态交互
- 可学习门控网络对每帧自适应分配专家权重
- 5 层交叉注意力 Transformer，16 头

**RAAV（Reliability-Aware Audio-Visual）**：
- 面部+行为特征通过 masked reliability-aware gating 帧级融合（可学习可靠性门控 + 模态先验）
- 音频通过可学习 bottleneck latent representations 提供辅助上下文（cross-attention）
- 非对称设计：视觉模态决定时序分辨率，音频提供窗口级补充

### 损失函数

基于 CCC 的混合损失，可选 MAE 项增强训练稳定性，valence/arousal 可独立加权

## 实验关键数据

| ID | 模型 | Valence | Arousal | Avg (dev) | Avg (test) |
|----|------|---------|---------|-----------|------------|
| 1 | Face: GRADA+Transformer | 0.587 | 0.651 | 0.619 | 0.54 |
| 3 | Behavior: Qwen3-multimodal+Mamba | 0.429 | 0.648 | 0.539 | – |
| 4 | Audio: WavLM+Pooling | 0.342 | 0.464 | 0.403 | – |
| 5 | Face+Audio (DCMMOE) | 0.625 | 0.667 | 0.646 | 0.58 |
| 7 | Face+Behavior+Audio (DCMMOE) | 0.610 | 0.688 | 0.649 | 0.61 |
| **8** | **Face+Behavior+Audio (RAAV)** | **0.608** | **0.707** | **0.658** | **0.62** |

**关键发现**：
- 多模态融合一致优于单模态（test CCC: 0.54→0.62）
- Qwen3 多模态嵌入（0.539）明显优于纯视觉嵌入（0.401），证实 VLM 行为嵌入的价值
- 所有模型中 arousal 估计比 valence 更可靠
- RAAV 融合策略在 arousal 上达到最高 0.7073，整体最优 0.6576
- 跨模态嘴部开合过滤有效提升音频模态质量

## 亮点

1. **VLM 行为嵌入的创新应用**：首次在 ABAW VA 任务中利用 Qwen3-VL 提取行为级语义嵌入，填补了 VLM 在连续情感估计中的空白
2. **两种互补融合策略**：DCMMOE 建模所有有序模态对交互，RAAV 采用非对称帧中心设计，各有优势
3. **跨模态音频过滤**：利用视觉线索（嘴部开合）过滤不可靠音频段，适应视频导向数据集特点
4. **系统性模态分析**：清晰展示了各模态贡献和融合增益

## 局限性

1. 作为竞赛论文（CVPR Workshop），方法较为工程导向，理论创新有限
2. Qwen3 行为嵌入需要手动设计 prompt，泛化性和鲁棒性待验证
3. 嘴部开合过滤是粗糙的语音存在近似，会遗漏或误判
4. 仅在 Aff-Wild2 单一数据集验证，跨数据集泛化未评估
5. 人脸检测需要手动身份校正，可扩展性受限

## 相关工作

- **单模态方法**：ResNet baseline (ABAW)、MobileViT、DDAMFN、MAE+CLIP+TCN (Zhou et al.)
- **多模态融合**：PDEM 音频特征 + 视觉骨干 (Dresvyanskiy et al.)、MAE+VGGish 集成 (Zhang et al.)、GR-JCA 门控 (Praveen et al.)、TAGF 时序门控 (Lee et al.)
- **赛事最优**：ResNet+VGGish+LogMel+TCN+cross-modal attention (Yu et al., 8th ABAW)
- 本工作是首个在 ABAW VA 任务中使用 VLM 行为嵌入的方法

## 评分

- 新颖性: ⭐⭐⭐ — VLM 行为嵌入是亮点，但整体为模块堆叠式工程方案
- 实验充分度: ⭐⭐⭐ — 模态消融充分，但仅单数据集，无与其他参赛队的横向对比
- 写作质量: ⭐⭐⭐ — 结构清晰但公式符号较多，部分细节冗长
- 价值: ⭐⭐⭐ — 为情感计算社区展示了 VLM 集成的可行路径，竞赛性能有竞争力
