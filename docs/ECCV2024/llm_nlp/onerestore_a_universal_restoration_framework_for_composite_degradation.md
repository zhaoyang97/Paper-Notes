# OneRestore: A Universal Restoration Framework for Composite Degradation

**会议**: ECCV 2024
**arXiv**: [2407.04621](https://arxiv.org/abs/2407.04621)
**代码**: [GitHub](https://github.com/gy65896/OneRestore)
**领域**: LLM/NLP
**关键词**: image restoration, composite degradation, scene descriptor, controllable restoration, contrastive loss

## 一句话总结
提出 OneRestore，一种基于 Transformer 的通用图像复原框架，通过场景描述符引导的交叉注意力机制和复合退化复原损失，能在单一模型中自适应地处理低光照、雾、雨、雪及其任意组合的复合退化场景，并支持文本/视觉双模式的可控复原。

## 研究背景与动机
1. **领域现状**: 图像复原研究已在单一退化场景（去雾、去雨、低光增强等）取得显著进展，但这些方法都是 One-to-One 模型，只能处理特定退化类型。
2. **现有痛点**:
   - 现实场景中多种退化因素常同时出现（如夜间雾中下雨），One-to-One 模型无法应对；
   - One-to-Many 部分参数共享方法（如 All-weather Net）需要为每种退化设置独立编码器，模型随退化种类线性增长；
   - One-to-Many 全参数共享方法（如 AirNet、TransWeather）直接混合训练，无法感知具体退化类型，可能去雾时反而增加噪声。
3. **核心矛盾**: 如何让单一模型既能识别复合退化的具体构成，又能按用户意图进行可控复原？
4. **本文解决什么**: 构建统一的复合退化成像模型 + 场景描述符引导的可控复原框架。
5. **切入角度**: 受人类标注流程启发——标注员需先了解退化类型才能评估质量，模型也应先"理解"退化场景再做复原。
6. **核心 idea**: 用场景描述嵌入作为退化"开关"，通过 cross-attention 引导 Transformer 对目标退化因素精准复原。

## 方法详解

### 整体框架
退化图像 $I(x)$ + 场景描述符 $e_t$ → Encoder（3 次下采样，每层包含 SDTB）→ Decoder（3 次上采样 + skip connection + 全局残差）→ 复原图像 $\hat{J}$

成像模型：$I(x) = \mathcal{P}_h(\mathcal{P}_{rs}(\mathcal{P}_l(J(x))))$，即清晰图像先经低光照→雨/雪→雾的级联退化。

### 关键设计

1. **复合退化成像模型（Composite Degradation Formulation）**:
   - 做什么：统一建模 4 类物理退化（低光照、雨、雪、雾）的级联过程
   - 核心思路：
     - 低光照：基于 Retinex 理论，$I_l(x) = \frac{J(x)}{L(x)} L(x)^\gamma + \varepsilon$，$\gamma \in [2,3]$
     - 雨：$I_{rs}(x) = I_l(x) + \mathcal{R}$
     - 雪：$I_{rs}(x) = I_l(x)(1-\mathcal{S}) + M(x)\mathcal{S}$
     - 雾：$I(x) = I_{rs}(x) \cdot t + A(1-t)$，$t = e^{-\beta d(x)}$
   - 设计动机：真实场景中退化是叠加的，需按物理规律建模级联关系。基于此构建了 CDD-11 数据集（11 类退化 + 清晰图），1383 张高分辨率图生成 13,013 训练对 + 2,200 测试对。

2. **场景描述符引导的 Transformer 块（SDTB）**:
   - 做什么：在每个 Transformer 块中融入场景退化信息，引导特征提取方向
   - 核心思路：包含 SDCA + SA + FFN 三个子模块。SDCA 使用场景描述嵌入生成 query，图像特征生成 key/value：
     $$\text{SDCA}(\mathbf{Q}_t, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}_t \cdot \mathbf{K}^\top}{\lambda}\right) \mathbf{V}$$
   - 设计动机：传统 self-attention 只在图像特征内部交互，无法利用退化类型先验。用场景描述生成 query 相当于"告诉"模型该关注哪种退化，实现从被动检测到主动引导的转变。

3. **场景描述符生成（Scene Descriptor Generation）**:
   - 做什么：提供两种模式生成场景描述嵌入——手动输入文本 vs 自动视觉属性提取
   - 核心思路：
     - 文本嵌入器：5 种基本场景文本经 GloVe → 12 种文本嵌入（含 7 种组合退化，由对应单退化嵌入取平均生成）→ MLP 精化
     - 视觉嵌入器：ResNet-18 提取视觉特征 → conv + dropout + linear → 视觉嵌入 → cosine similarity 匹配最相似文本嵌入
     - 训练用 cosine cross-entropy loss：$S(e_v, e_t) = \frac{e^{\cos(e_v, e_t)}}{\sum_{t_i=1}^{N_t} e^{\cos(e_v, e_{t_i})}}$
   - 设计动机：文本嵌入提供精确可控的场景描述（精度更高），视觉嵌入提供自动化能力（97.55% 准确率），两者互补。

4. **复合退化复原损失（CDRL）**:
   - 做什么：在传统对比损失基础上引入多个退化负样本，增强模型区分能力
   - 核心思路：
     $$\mathcal{L}_c = \sum_{k=1}^{K} \xi_k \frac{\mathcal{L}_1(V_k(J), V_k(\hat{J}))}{\xi_c \mathcal{L}_1(V_k(\hat{J}), V_k(I)) + \sum_{o=1}^{O} \xi_o \mathcal{L}_1(V_k(I_o), V_k(\hat{J}))}$$
     使用 VGG-16 的 3、8、15 层特征，$O=10$ 个其他退化负样本
   - 设计动机：传统对比损失只用输入退化图作负样本，可能把复原结果推向其他退化形式。CDRL 同时远离所有退化类型。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \alpha_1 \mathcal{L}_1^s + \alpha_2 \mathcal{L}_M + \alpha_3 \mathcal{L}_c$（smooth $l_1$ + MS-SSIM + CDRL）
- 分两阶段训练：先训 text/visual embedder（200 epochs, lr=0.0001），再训 OneRestore（120 epochs, lr=0.0002）
- 训练图裁为 256×256 patch，stride 200，随机翻转，生成 312k 训练对
- 8 张 NVIDIA L40 GPU

## 实验关键数据

### 主实验（CDD-11 数据集）
| 方法 | 类型 | PSNR↑ | SSIM↑ | 参数量 |
|------|------|-------|-------|--------|
| Restormer | One-to-One | 26.99 | 0.8646 | 26.13M |
| SRUDC | One-to-One | 27.64 | 0.8600 | 6.80M |
| WGWSNet | One-to-Many | 26.96 | 0.8626 | 25.76M |
| PromptIR | One-to-Many | 25.90 | 0.8499 | 38.45M |
| **OneRestore** | One-to-Composite | **28.47** | **0.8784** | **5.98M** |
| **OneRestore†** | One-to-Composite | **28.72** | **0.8821** | 5.98M |

### 消融实验
| 配置 | PSNR↑ | SSIM↑ | 可控性 |
|------|-------|-------|--------|
| FFN only | 24.81 | 0.8607 | ✗ |
| SA + FFN | 27.19 | 0.8697 | ✗ |
| SDCA + FFN | 27.93 | 0.8767 | ✓ |
| **SDCA + SA + FFN** | **28.72** | **0.8821** | **✓** |

| 损失组合 | PSNR↑ | SSIM↑ |
|---------|-------|-------|
| Smooth $l_1$ only | 28.16 | 0.8633 |
| Smooth $l_1$ + MS-SSIM | 27.54 | 0.8708 |
| Smooth $l_1$ + MS-SSIM + CL | 27.61 | 0.8723 |
| **Smooth $l_1$ + MS-SSIM + CDRL** | **28.72** | **0.8821** |

### 关键发现
- SDCA 模块单独加入即可带来 3.12 dB 的提升（vs FFN only），且赋予模型可控性
- CDRL 比传统 CL 在 PSNR 上提升 1.11 dB，SSIM 提升 0.01
- 视觉嵌入器的场景识别准确率达 97.55%，误判主要发生在退化因素不显著时
- 文本嵌入器 > 视觉嵌入器 > Classifier，因为固定数量的文本描述符能更好地充当"退化开关"
- 在真实场景中也展现了良好的泛化能力

## 亮点与洞察
- **One-to-Composite 范式**: 首次提出系统化处理复合退化的框架，区别于 One-to-One 和 One-to-Many
- **可控复原**: 通过手动输入不同文本描述，可选择性地只去除特定退化（如只去雾不去雨），在复原领域罕见
- **极小参数量**: 仅 5.98M 参数即超越 26-38M 的竞品，说明"让模型知道做什么"比"堆参数"更重要
- **CDRL 损失**: 用多退化负样本构建更严格的下界约束，思路可迁移到其他多任务学习场景
- **级联退化建模**: 按物理规律建模低光→雨/雪→雾的级联关系，比简单叠加更物理合理

## 局限性 / 可改进方向
- 对极高密度退化场景（如暴雨+浓雾）效果受限
- 未考虑的退化类型（如运动模糊、压缩伪影）出现时预测能力受限
- 视觉嵌入器的 97.55% 准确率仍有误判空间，可能导致错误复原方向
- 场景描述嵌入的组合方式（取平均）较简单，可探索更强的组合策略

## 相关工作与启发
- **vs AirNet / TransWeather**: 全参数共享的 One-to-Many 方法无法感知退化类型，混合训练时不同退化间互相干扰
- **vs All-weather Net (Li et al.)**: 部分参数共享，每种退化需独立编码器，扩展性差
- **vs Restormer**: 虽然 Restormer 是强 baseline，但在复合退化上 OneRestore 超出 1.73 dB，说明退化感知的重要性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统定义复合退化问题并提出 One-to-Composite 范式，场景描述符控制复原方向新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 自建 CDD-11 数据集、14 种 SOTA 对比、多维消融、真实场景验证、可控性展示齐全
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，问题定义明确，但公式和符号略多
- 价值: ⭐⭐⭐⭐⭐ 解决了真实场景复合退化的核心痛点，可控复原开创了新方向，参数效率极高
