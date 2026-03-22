# Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction

**会议**: CVPR2025  
**arXiv**: [2603.11206](https://arxiv.org/abs/2603.11206)  
**代码**: 待确认  
**领域**: medical_imaging  
**关键词**: breast tumor segmentation, vision-language, evidential learning, cross-attention, DCE-MRI

## 一句话总结

提出 TextBCS 模型，通过阶段分割的视觉-语言交互模块（SVLI）和证据学习（EL）策略，利用文本提示辅助乳腺肿瘤分割，在 Duke-Breast-Cancer-MRI 数据集上 Dice 达 85.33%，超越所有对比方法。

## 研究背景与动机

乳腺癌是全球女性最常见的癌症死因之一。DCE-MRI 因其高灵敏度被广泛用于乳腺肿瘤检测，但存在两大挑战：(1) 肿瘤区域与正常组织之间对比度低，难以精确定位肿瘤轮廓；(2) 边界模糊导致分割不确定性高。现有方法仅依赖图像模态学习，缺乏语义层面的指导信息。文本提示可以提供关于病变区域的位置、形状、大小等先验知识，有助于解决低对比度场景下的定位问题。然而，现有文本引导方法仅进行浅层的图文交互，限制了有效的跨模态对齐与融合。

## 方法详解

### 整体框架

TextBCS 基于 UNet 架构，包含两个核心创新模块：

1. **SVLI（Stage-divided Vision-Language Interaction）**：在编码器每个下采样阶段执行视觉-语言双向交互
2. **EL（Evidential Learning）**：在解码器端进行像素级不确定性估计

文本输入使用 BioClinicalBERT 编码为文本嵌入。

### SVLI 模块设计

**(1) 阶段分割双向交叉注意力机制**：

在每个下采样阶段 $s$ 进行两轮交互：
- **Vision Query 模块**：视觉特征作为 Query，文本特征作为 Key/Value，通过多头交叉注意力生成文本感知的视觉特征 $F_V^s$
- **Language Query 模块**：文本特征作为 Query，文本感知的视觉特征作为 Key/Value，生成视觉感知的文本特征 $F_L^s$

每个模块内部包含两层交叉注意力 + FFN + 残差连接。

**(2) 阶段分割跨模态对齐损失**：

在每个特征层级计算文本到图像的对比损失：
$$\mathcal{L}_{con}^{sj} = \begin{cases} -\log(\sigma(\text{Sim}(F_V^{s,j}, F_L^{s,j})/\tau_s)) & j \in Z^+ \\ -\log(1-\sigma(\text{Sim}(F_V^{s,j}, F_L^{s,j})/\tau_s)) & j \in Z^- \end{cases}$$

区别于先前方法仅在最终特征层做对齐，SVLI 确保低层和高层特征均进行跨模态对齐。

### 证据学习模块

在解码器输出后插入 Softplus 激活函数获得非负证据 $e = [e_1, ..., e_C]$，构建 Dirichlet 分布 $Dir(p|\alpha)$（其中 $\alpha = e + 1$）来建模分割概率的分布。

- **信念质量**：$b_{i,j}^c = e_{i,j}^c / W$
- **不确定性**：$u_{i,j} = C / W$（$W = \sum_c \alpha_{i,j}^c$）

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{Dice} + \lambda_1 \mathcal{L}_{ice} + \lambda_2 \mathcal{L}_{KL} + \lambda_3 \mathcal{L}_{con}$$

- $\mathcal{L}_{ice}$：基于 Dirichlet 的积分交叉熵损失
- $\mathcal{L}_{KL}$：KL 散度正则项，确保错误类别证据较低
- $\lambda_1 = 10^{-3}$, $\lambda_2 = 5e\text{-}7 \cdot \min\{1, n_{epoch}/100\}$（渐进增大）, $\lambda_3 = 10^{-3}$

## 实验关键数据

**主实验（Duke-Breast-Cancer-MRI 数据集，922 例患者，3876 切片）**：

| 方法 | 文本 | Dice (%) | mIoU (%) | Param (M) |
|------|------|----------|----------|-----------|
| UNet | ✗ | 81.54 | 73.22 | 14.8 |
| TransUNet | ✗ | 83.14 | 75.49 | 105 |
| MGCA | ✓ | 84.28 | 75.44 | 135.6 |
| LViT | ✓ | 82.79 | 73.21 | 29.7 |
| **TextBCS** | ✓ | **85.33** | **76.08** | 32.5 |

**消融实验**：

| Baseline | SVLI | EL | Dice (%) |
|----------|------|----|----------|
| ✓ | | | 81.54 |
| ✓ | ✓ | | 84.41 (+2.87) |
| ✓ | | ✓ | 83.19 (+1.65) |
| ✓ | ✓ | ✓ | **85.33** (+3.79) |

- 所有对比方法的 t 检验 p 值均 < 0.05，统计显著
- 模型对文本提示风格变化具有鲁棒性

## 亮点

- 首个将文本引导应用于 DCE-MRI 乳腺肿瘤分割的方法
- SVLI 在编码器每个阶段进行视觉-语言交互，比仅在最终层或跳跃连接处交互更充分
- 证据学习提供像素级不确定性量化，对模糊边界给出高不确定性而非过度自信预测
- 参数量（32.5M）和 FLOPs（52.3G）在文本引导方法中最低，效率高
- 可解释性研究表明 SVLI 有效引导模型关注癌变区域

## 局限性

- 文本提示需要放射科医生手动提供，实际部署受限（虽讨论了 LLM 自动生成策略但未验证）
- 仅在单个公开数据集上验证，泛化能力待验证
- 文本提示格式较简单（location/shape/size/number），未充分利用更丰富的临床描述
- 错误/不充分的文本提示会导致错误分割（如 Fig. 5 所示）
- 仅处理 2D 切片，未利用 3D 体积信息

## 相关工作

- **LViT** (Li et al.)：在编码器下采样阶段集成文本，本文方法在此基础上增加双向交叉注意力
- **MGCA**：利用文本引导但缺乏充分的图文交互，且不评估分割可靠性
- **CLIP/GLoRIA/ConVIRT**：使用对比学习进行图文对齐，但交互深度不够
- **EDL** (Sensoy et al.)：本文 EL 模块的理论基础，通过 Dirichlet 分布建模预测不确定性

## 评分

- 新颖性: ⭐⭐⭐⭐ (SVLI + EL 组合设计合理，文本引导乳腺分割属首创)
- 实验充分度: ⭐⭐⭐ (消融充分但仅单数据集，缺跨域验证)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，公式推导完整)
- 价值: ⭐⭐⭐⭐ (文本引导+不确定性估计组合在临床场景有实际意义)
