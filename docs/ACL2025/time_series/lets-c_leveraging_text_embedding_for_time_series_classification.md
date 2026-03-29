# LETS-C: Leveraging Text Embedding for Time Series Classification

**会议**: ACL 2025
**arXiv**: [2407.06533](https://arxiv.org/abs/2407.06533)
**代码**: 无（使用 OpenAI API）
**领域**: 时间序列
**关键词**: time series classification, text embedding, cross-modal transfer, lightweight model, CNN-MLP

## 一句话总结
提出 LETS-C——将时间序列数字化为文本字符串后用 text embedding 模型编码，与原始时间序列元素级相加融合后送入轻量 CNN+MLP 分类头，在 UEA 10 个多变量时间序列数据集上以仅 14.5% 的可训练参数量超越 OneFitsAll（GPT-2 微调）等 27 个 baseline 达到 SOTA。

## 研究背景与动机

1. **领域现状**：时间序列分类是金融/医疗/活动识别等领域的关键任务。最近 LLM 微调方法（如 OneFitsAll 基于 GPT-2）在标准 benchmark 上达到 SOTA，但需要数百万可训练参数。
2. **现有痛点**：LLM 微调方法 (a) 模型巨大（即使冻结大部分参数仍需百万级可训练参数）；(b) 推理成本高；(c) 对资源受限场景不实用。
3. **核心矛盾**：能否利用语言模型在序列建模上的成功，但避免微调 LLM 的高成本？
4. **本文要解决什么**：用 text embedding 模型（而非 LLM 微调）编码时间序列，配合轻量分类头实现高效分类。
5. **切入角度**：text embedding 模型已在 MTEB 上展示强大的序列表示能力，且推理是一次性的（可预计算存储），比 LLM 微调高效得多。
6. **核心 idea**：时间序列 → 数字字符串 → text embedding → 与原始序列融合 → CNN+MLP 分类。

## 方法详解

### 整体框架
输入：多变量时间序列 $\mathbf{x}_i \in \mathbb{R}^{d \times l_x}$。步骤：(1) min-max 归一化 → (2) 每个维度格式化为 digit-space 文本串 → (3) text-embedding-3-large 编码为 $\mathbf{e}_i \in \mathbb{R}^{d \times l_e}$ → (4) embedding 与时间序列元素级相加（zero-padding 对齐维度）→ (5) 1D CNN + MLP 分类。

### 关键设计

1. **Digit-Space Tokenization**
   - 做什么：将浮点数转为每位数字独立 tokenize 的字符串
   - 核心思路：`0.645, 6.45` → `"6 4 , 6 4 5"`，逗号分隔时间步，空格分隔数字位
   - 设计动机：BPE 等子词 tokenizer 会任意切割数字（Gruver et al., 2024），导致相似数字表示差异大。Digit-space 确保每位独立 tokenize，保持数值完整性

2. **Text Embedding 模型编码**
   - 做什么：用 OpenAI text-embedding-3-large（3072维）编码格式化文本串
   - 核心思路：每个维度独立编码（channel-wise），得到 $d \times l_e$ 的 embedding 矩阵。**Embedding 是一次性预计算**，可存储复用
   - 设计动机：text embedding 模型经大规模文本训练，对序列模式有内在理解能力；且推理是 API 调用无需 GPU

3. **元素级加法融合**
   - 做什么：$\text{fused} = \text{embedding} + \text{timeseries}$（zero-padding 对齐维度）
   - 设计动机：类似 ResNet 的 shortcut connection——embedding 提供高级语义特征，原始时间序列保留精确数值信息。实验证明加法优于拼接或注意力融合

4. **轻量 CNN+MLP 分类头**
   - 做什么：1D CNN 提取局部模式 → 展平 → MLP + softmax 分类
   - 设计动机：**刻意简单**——验证 text embedding 本身已提供足够强的特征表示，不需要复杂模型

## 实验关键数据

### 主实验 (UEA 10 数据集平均)

| 方法 | 平均准确率 | 可训练参数 | AvgWins% |
|------|-----------|-----------|---------|
| DTW | 66.97% | - | 0% |
| TimesNet | 73.60% | ~1.5M | 10% |
| PatchTST | 74.33% | ~1.2M | 20% |
| OneFitsAll (GPT-2) | 75.20% | ~1.0M | 40% |
| MOMENT | 76.50% | ~0.8M | 30% |
| **LETS-C** | **78.56%** | **~0.14M** | **90%** |

### 参数效率对比

| 方法 | 可训练参数 | vs LETS-C |
|------|-----------|----------|
| OneFitsAll | ~1.0M | 6.9× |
| TimesNet | ~1.5M | 10.3× |
| **LETS-C** | **~0.14M** | **1×** |

### 消融实验

| 配置 | 平均准确率 |
|------|-----------|
| 仅 text embedding（无原始TS）| 73.2% |
| 仅原始 TS（无 embedding）| 71.8% |
| 拼接融合 | 76.1% |
| **加法融合（LETS-C）** | **78.56%** |

### 关键发现
- **Text embedding 的同类内聚力**：同类时间序列的 text embedding 余弦相似度显著高于异类——说明 text embedding 天然具备时间序列的判别能力
- **LETS-C 在 10 个数据集中 9 个排名 top-2**——泛化性极强
- **参数量仅为 SOTA 的 14.5%**——在 JPMorgan 等工业场景有直接部署价值
- **多种 text embedding 模型都有效**：不仅 OpenAI，GTE/Mistral-embed 等开源模型也能达到超越 OneFitsAll 的性能
- **模型压缩友好**：embedding 维度从 3072 截断到 768 仅损失 <2% 准确率

## 亮点与洞察
- **"text embedding 理解时间序列"的惊人发现**：文本编码器从未见过时间序列数据，但 digit-space 格式化后的数值字符串 embedding 竟然具有强判别力——暗示语言模型对数字序列模式有某种隐含理解能力
- **极致简洁的设计**：方法本质就是"把数字变成字符串 → 调 API → 加回去 → 跑个小网络"——但效果超越所有复杂方法
- **一次性预计算 embedding**：无需 GPU 训练 embedding，API 调用后存储复用——部署极简
- **挑战"LLM 微调必要性"**：不需要接入 Transformer 内部，只用 embedding 接口就够了

## 局限性 / 可改进方向
- **依赖商业 API**：OpenAI text-embedding-3-large 是付费 API，成本和可控性受限
- **仅多变量分类**：未扩展到预测（forecasting）、异常检测等其他时间序列任务
- **无验证集**：benchmark 设置与先前工作一致（train/test only），报告的是 upper bound
- **为什么 text embedding 对时间序列有效？**缺乏机制性解释——数字序列的模式是否被 BPE 训练隐式学到？
- **长时间序列受限**：受 token limit（8191）约束

## 相关工作与启发
- **vs OneFitsAll (Zhou et al., 2024)**：微调 GPT-2 全部参数，准确率低于 LETS-C 且参数量多 7x
- **vs MOMENT (Goswami et al., 2024)**：时间序列预训练 foundation model，自监督学习表示；LETS-C 用 text embedding 零训练获取表示
- **vs TS2Vec/TNC**：对比学习方法需要领域数据预训练，LETS-C 直接利用语言模型的"跨域"表示

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 text embedding 用于时间序列分类，idea 极简但效果惊人
- 实验充分度: ⭐⭐⭐⭐⭐ 27 baseline + 10 数据集 + 多 embedding 模型 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，Figure 1 直观
- 价值: ⭐⭐⭐⭐⭐ 为时间序列分析开辟了全新的"语言模型作为特征提取器"范式
