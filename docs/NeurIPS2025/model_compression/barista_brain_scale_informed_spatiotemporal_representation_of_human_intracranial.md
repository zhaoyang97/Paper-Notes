# BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG

**会议**: NeurIPS 2025  
**arXiv**: [2512.12135](https://arxiv.org/abs/2512.12135)  
**代码**: https://github.com/ShanechiLab/BaRISTA  
**领域**: 神经科学 / 基础模型  
**关键词**: 颅内脑电, 时空Transformer, 空间编码尺度, 掩码重建, 预训练

## 一句话总结
BaRISTA 系统探索 iEEG Transformer 的空间编码尺度（电极/脑区/脑叶），发现脑区级编码 + 空间掩码重建在语言任务解码上达 86.2% AUC（vs PopT 79.5%），编码尺度选择的影响 > 掩码策略选择，且跨被试泛化性好。

## 研究背景与动机

1. **领域现状**：iEEG 提供高时空分辨率脑活动，Transformer 预训练模型（PopT, Brant）已用于 iEEG 但空间编码选择未被系统研究。
2. **现有痛点**：不同患者电极位置完全不同，channel 级编码无法跨患者泛化。脑区级和脑叶级编码是否更好？空间编码和掩码策略如何交互？
3. **核心矛盾**：精细空间分辨率（channel）提供最多信息但跨患者不一致；粗空间分辨率（脑叶）跨患者一致但可能丢失局部信息。
4. **本文要解决什么？** 系统对比三种空间编码尺度，找到最优的空间编码-掩码组合。
5. **切入角度**：将 channel/atlas parcel/lobe 三种空间粒度作为实验变量，在掩码重建预训练框架下系统消融。
6. **核心 idea 一句话**：在 iEEG Transformer 中系统对比三种空间编码尺度，发现脑区级（atlas parcel）是最优的空间粒度——兼顾跨患者一致性和局部信息保留。

## 方法详解

### 整体框架
iEEG 数据（2048Hz）→ **时间分词**（Dilated CNN 提取 250ms patch）→ **空间编码**（channel/atlas/lobe 三种粒度的可学习嵌入 $E_j$）→ token $S_{ij} = B_{ij} + E_j$ → **空间掩码**（随机选掩码空间类别）→ **Transformer**（12 层/4 头/d=64 + RoPE）→ **EMA 目标重建**（online 编码器 vs EMA target 编码器的 MSE）

### 关键设计

1. **三种空间编码尺度**:
   - Channel: (x,y,z) MNI 坐标 → 可学习嵌入（最精细但跨患者不一致）
   - Atlas parcels: Destrieux atlas 脑区划分（中等粒度，跨患者一致）
   - Lobes: 脑叶 + 皮层下区域（最粗但最稳定）

2. **空间掩码重建预训练**:
   - 随机选择空间类别进行掩码（如掩盖某个脑区所有电极的 patch）
   - Online tokenizer $\mathcal{F}$ 和 EMA target tokenizer $\tilde{\mathcal{F}}$（动量 0→0.996 warmup）
   - $\mathcal{L} = \frac{1}{|B_{target}|}\sum \|\tilde{B}_{ij} - \hat{B}_{ij}\|_2^2$

3. **交错时空序列**: 空间和时间 token 交错排列，使 attention 同时处理时空依赖

### 损失函数 / 训练策略
- Brain Treebank 数据集：10 位癫痫患者，26 sessions，2048 Hz
- 预训练 + 线性探测评估

## 实验关键数据

### 主实验（下游分类 AUC %）

| 编码尺度/掩码 | 句子起始 | 语音/非语音 |
|------------|---------|-----------|
| Channel | 77.8% | 76.4% |
| **Parcel** | **86.2%** | **86.9%** |
| Lobe | 84.2% | 84.1% |
| PopT baseline | 79.5% | 77.5% |
| Brant baseline | 76.7% | 69.1% |

### 消融（ANOVA）

| 因素 | p 值 | 效应量 |
|------|------|--------|
| 编码尺度 | **p<1e-3** | 大 |
| 掩码策略 | p=0.01-0.04 | 中 |
| 交互效应 | Channel 编码 + Channel 掩码最佳匹配 | — |

### 关键发现
- Parcel 编码显著优于 Channel（+8.4% 句子起始，+10.5% 语音）——解剖先验比精确坐标更重要
- 编码尺度的影响 > 掩码策略——选对编码比选对掩码更关键
- 跨被试泛化：hold-out 被试仍达 84.1%（vs 86.9% with target），说明脑区编码确实帮助跨患者泛化
- 数据规模正相关：5%→75% 预训练数据，性能持续提升

## 亮点与洞察
- **空间编码尺度是被忽视的关键设计选择**：之前的工作默认用 channel 级，BaRISTA 证明脑区级更优
- **"太精细不如适中"**：channel 编码虽然信息最多但跨患者不一致导致泛化差，atlas parcel 平衡了精度和泛化

## 局限性 / 可改进方向
- 仅基于解剖学划分，未探索功能性脑区编码
- 仅做空间掩码，未测试时空联合掩码
- 单一采样率（2048 Hz）
- Dilated CNN 时间编码可能不是最优

## 相关工作与启发
- **vs PopT**: PopT 用 channel 编码，BaRISTA 证明 parcel 更优（+6.7%）
- **vs Brant**: Brant 用脑区编码但无系统消融，BaRISTA 提供完整分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统消融空间编码尺度是首次
- 实验充分度: ⭐⭐⭐⭐⭐ 三尺度×多掩码×ANOVA+跨被试+数据规模
- 写作质量: ⭐⭐⭐⭐ 实验设计严谨
- 价值: ⭐⭐⭐⭐ 为 iEEG 基础模型设计提供了关键指导
- 脑电信号的空间层级结构需要多尺度建模——粗粒度+细粒度互补
- 更大空间尺度（lobe级）改善解码性能，预训练显著提升少样本场景
- 该方法的核心创新在于设计思路的简洁性和有效性
- 实验结果充分验证了核心假设

