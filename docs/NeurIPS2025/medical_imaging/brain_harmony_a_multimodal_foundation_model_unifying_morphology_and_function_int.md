# Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens

**会议**: NeurIPS 2025  
**arXiv**: [2509.24693](https://arxiv.org/abs/2509.24693)  
**代码**: [hzlab/Brain-Harmony](https://github.com/hzlab/Brain-Harmony)  
**领域**: medical_imaging  
**关键词**: 脑影像基础模型, 多模态融合, fMRI, sMRI, 1D token  

## 一句话总结

首个统一脑结构形态（T1 sMRI）与功能动态（fMRI）的多模态脑基础模型，通过几何谐波预对齐和时序自适应 Patch Embedding（TAPE）将高维神经影像压缩为紧凑的 1D token 表示，在神经发育/退行性疾病诊断和认知预测任务上全面超越先前方法。

## 研究背景与动机

1. **单模态局限**：现有脑基础模型要么只建模结构（BrainMVP）要么只建模功能（BrainLM, Brain-JEPA, BrainMass），无法同时捕获结构-功能的互补信息。
2. **结构约束功能**：神经科学研究表明脑功能活动以皮层几何形态为约束传播（"function follows structure"），但已有功能模型完全忽略这一先验。
3. **异质 TR 问题**：不同扫描仪/协议的 fMRI 采样率（TR）各异，BrainLM 和 Brain-JEPA 只能处理固定 TR，严重限制了多数据集联合预训练和临床部署。
4. **信息丢失**：BrainMass 基于静态功能连接，丢弃了 BOLD 信号中的非平稳动态（状态切换、共激活模式演化等）。
5. **数据规模**：预训练使用了迄今最大规模的神经影像数据集——64,594 张 T1 3D 体积（~1400 万张图像）+ 70,933 条 fMRI 时间序列。
6. **压缩表示需求**：需要将高维神经影像数据深度压缩为紧凑、信息密集的 1D 连续 token，形成人脑的统一潜在空间。

## 方法详解

### 整体框架

BrainHarmonix 采用两阶段预训练：**单模态编码（UE）** → **多模态融合（MF）**。

- **BrainHarmonix-S**（结构）：ViT-B 骨干 + 3D Masked Autoencoder（MAE），在 64,594 张 T1 图像上预训练。
- **BrainHarmonix-F**（功能）：ViT-B 骨干 + JEPA 框架，在 252,961 条 fMRI 时序上预训练。
- **Harmonizer**：ViT-B 编码器 + MAE 风格解码器，通过可学习的 1D brain hub token 融合两个模态的潜在表示。

### 三个核心设计

1. **几何谐波预对齐**（Geometric Harmonics Pre-alignment）
   - 基于群体级皮层表面网格计算 Laplace-Beltrami 算子（LBO）的特征值分解，得到一组正交的几何谐波 $\psi_i$。
   - 将几何谐波下采样到 ROI 级别（$\bar{\psi}_i \in \mathbb{R}^{N\times 1}$），通过线性层投影为 Transformer 的位置编码。
   - 将物理先验（皮层形态约束）注入 fMRI 表示，增强跨被试/跨数据集的对齐能力。

2. **时序自适应 Patch Embedding（TAPE）**
   - 定义统一的时间窗口 $\tau$，根据 TR 动态计算 patch 大小：$k = \text{round}(\tau / s)$。
   - 通过伪逆缩放（PI-resize）自适应调整嵌入权重：$\omega = ((B_k^{k^*})^T)^\dagger \cdot \omega^*$。
   - 不同长度的时序用 zero-padding + attention mask 处理，首次实现任意 TR 的 fMRI 预训练。
   - 衍生出首个针对 fMRI 时间序列的数据增强方法：将高分辨率扫描降采样到多级 TR（例如 UKB 0.735s → 1.47s/2.205s/2.94s）。

3. **Brain Hub Token 融合**
   - 引入 $N_H=128$ 个可学习的连续 1D brain hub token $\mathbf{H}_0 \in \mathbb{R}^{N_H \times d}$。
   - 将 hub token 与结构/功能 token 拼接输入 Harmonizer：$\mathbf{Z}_0 = [\mathbf{H}_0; \mathbf{Z}_S; \mathbf{Z}_F]$。
   - 自注意力机制使 hub token 同时聚合两个模态的信息，并促进跨模态交互。

### 损失函数

- **UE 阶段**：结构用 MAE 重建损失，功能用 JEPA 预测损失（标准实现）。
- **MF 阶段**：双模态重建 MSE 损失：

$$\mathcal{L}_{\text{fusion}} = \|\mathcal{D}_S(\tilde{\mathbf{H}}) - \mathbf{Z}_S\|_2^2 + \|\mathcal{D}_F(\tilde{\mathbf{H}}) - \mathbf{Z}_F\|_2^2$$

## 实验

### 数据集

| 阶段 | 数据集 | 规模 |
|------|--------|------|
| 预训练 T1 | UKB + ABCD | 64,594 扫描 |
| 预训练 fMRI | UKB + ABCD（含 TR 增强） | 252,961 条时序 |
| 融合 | UKB + ABCD 配对 | 69,360 对 |

### 主要结果

**神经发育障碍诊断（Table 1）**：

| 模型 | ABIDE-I ACC/F1 | ABIDE-II ACC/F1 | ADHD-200 ACC/F1 |
|------|---------------|-----------------|-----------------|
| BrainMass | 65.64/69.07 | 59.35/71.86 | 65.99/61.27 |
| Brain-JEPA | 无法测试（固定TR） | — | — |
| BrainHarmonix-F | 57.39/71.24 | 62.90/72.76 | 67.69/68.75 |
| **BrainHarmonix** | **63.13/72.63** | **66.67/74.88** | **70.09/66.72** |

**神经退行性疾病 & 认知预测（Table 2）**：

| 模型 | PPMI ACC/F1 | ADNI ACC/F1 | HCP-A MAE/ρ |
|------|------------|------------|-------------|
| Brain-JEPA | 60.36/48.76 | 59.60/60.78 | 5.62/0.26 |
| BrainHarmonix-F | 62.79/52.90 | 61.62/64.80 | 5.77/0.30 |
| **BrainHarmonix** | **64.34/56.40** | **64.65/68.75** | 6.56/**0.42** |

### 关键发现

1. **多模态融合持续带来增益**：BrainHarmonix 在 6 个基准中的 5 个上取得最佳结果（p<0.05 显著性）。
2. **Token 缩放**：1D token 数从 32 增至 256 时性能稳步提升，128→256 时趋于饱和。
3. **线性探测即超越 SOTA**：仅 0.0015M 可训练参数的线性探测已超越先前最优方法。
4. **消融验证**：几何谐波预对齐和 TR 数据增强均带来显著且一致的提升。
5. **注意力分析**：128 个 hub token 中 93 个专注 fMRI、30 个专注 T1、5 个跨模态，展现了自动分化的模态偏好。

## 亮点

- **首个结构-功能统一的脑基础模型**，从问题定义到技术方案都有明确的神经科学动机。
- **TAPE 设计优雅**：一个简洁的公式即解决了异质 TR 这一领域长期痛点，并由此衍生出 fMRI 数据增强。
- **Brain hub token 的信息瓶颈思想**令人信服——将复杂的脑结构和功能压缩为 128 个 1D token 仍能有效重建。
- **可解释性分析有深度**：注意力模式与 ASD 相关的脑区/网络高度吻合（颞顶叶、默认模式网络等）。
- 预训练仅需 8 张 H100，融合阶段约 10 小时，计算成本对领域而言合理。

## 局限性

- 预训练数据年龄分布偏中老年（UKB 44-83 岁）和儿童（ABCD 8-11 岁），婴幼儿和青年段缺失。
- 单模态编码器与融合模块分别训练，未端到端联合优化，可能存在表示不一致。
- HCP-A 认知预测上 MAE 指标 BrainHarmonix（6.56）不如 BrainMVP2（5.39），说明多模态融合并非所有指标都优。
- 仅在 Schaefer-400 分区上验证，未探索更细/更粗粒度或无分区方案的影响。
- 目前仅涉及静息态 fMRI，未扩展到任务态 fMRI 或其他模态（DTI/EEG）。

## 相关工作

| 模型 | 模态 | 异质TR | 预训练策略 |
|------|------|--------|-----------|
| BrainLM | fMRI 时序 | ✗ | MAE |
| Brain-JEPA | fMRI 时序 | ✗ | JEPA |
| BrainMass | fMRI 连接 | ✓（静态） | 自监督 |
| BrainMVP | T1/T2 结构 | N/A | 多参数 MRI 对比 |
| BDO | fMRI 动态 | — | 随机最优控制 |
| **BrainHarmonix** | **T1 + fMRI** | **✓** | **MAE + JEPA + Hub Token 融合** |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多模态脑基础模型，TAPE 和 hub token 设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 6 个下游基准 + 消融 + token 缩放 + 可解释性分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表精美，神经科学动机阐述到位
- 价值: ⭐⭐⭐⭐ — 为脑影像领域建立了多模态基础模型范式，但具体性能提升幅度有限
