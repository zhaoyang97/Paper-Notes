# BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation

**会议**: ECCV2024  
**arXiv**: [2408.05926](https://arxiv.org/abs/2408.05926)  
**代码**: [hee-suk-yoon/BI-MDRG](https://github.com/hee-suk-yoon/BI-MDRG)  
**领域**: multimodal_vlm  
**关键词**: 多模态对话, 图像一致性, 视觉语言模型, 文本到图像生成, 对话响应生成

## 一句话总结

提出 BI-MDRG 框架，通过桥接图像历史信息来增强多模态对话中文本回复的图像 grounding 能力和连续图像回复中物体的一致性。

## 研究背景与动机

多模态对话响应生成 (MDRG) 要求模型根据对话上下文生成文本、图像或混合形式的回复。由于缺少大规模多模态对话数据集，之前的方法（如 Divter）采用**文本作为中间媒介**：先将对话历史中的图像转换为文本描述，再基于纯文本生成回复，最后通过 text-to-image 模型将文本描述转为图像。

这种"绕路"策略存在两个核心问题：

1. **文本回复缺乏图像 grounding**：模型只能通过图像的文字描述（如"一条狗在吃西瓜"）来理解图像，无法回答需要实际视觉信息的问题（如"你的狗是什么品种？"）
2. **图像回复物体不一致**：在多轮对话中生成的图像无法保持同一物体的视觉一致性（如前后两轮中的"狗"长得完全不同）

## 方法详解

### 整体框架

BI-MDRG 由三个核心部分组成：

1. **Textual Dialogue Response Generator $\mathcal{G}$**：基于 decoder-only 语言模型，新增视觉交叉注意力层，可直接接收 Visual Encoder $\mathcal{V}$ 提取的图像特征
2. **Citation Module $\mathcal{C}$**：无需训练的模块，利用现成组件（POS tagger + 目标检测 + 分割 + 特征提取）追踪对话中反复出现的同一物体，为文本图像描述添加引用标签（如 `[cite]0[/cite]`）
3. **Customized Text-to-Image Model $\mathcal{F}$**：推理时根据引用标签将历史图像中的同一物体输入定制化图像生成模型，维持物体一致性

整体初始化自 **OpenFlamingo 4B**（ViT-L + RedPajama 3B），图像描述由 **BLIP2-flan-t5-xl** 生成。

### 关键设计

**1. 多模态因果注意力掩码调制（Multimodal Causal Attention Mask Modulation）**

传统因果掩码允许文本回复 $r_i^{\text{Text}}$ 访问前面所有文本图像描述 $u_{1:i-1}'$，导致模型依赖文本描述而非真实图像。BI-MDRG 修改注意力掩码，**屏蔽文本回复对前序图像描述的访问**，强制模型通过交叉注意力层直接从图像特征获取视觉信息。同时保留图像描述作为输入，因为模型仍需生成文本图像描述以供 text-to-image 模型使用。

**2. Citation Module 引用追踪**

采用 pipeline 方式，全部使用预训练现成组件，无需在目标数据集上训练：

- **spaCy** 做 POS 标注，定位图像描述中的主要物体词 $o_i$
- **GroundingDINO** 做开放集目标检测，获取物体 bounding box
- **SAM** 做图像分割，生成物体掩码
- **DINOv2** 提取去除背景后的物体视觉特征 $f_i$

最后基于余弦相似度（阈值 $\tau=0.6$）对特征聚类，将属于同一物体的不同出现赋予相同的引用编号。例如"a dog is in front of a fireplace"和"a dog running in the snow"中的"dog"若特征相似，均标注为 `dog[cite]0[/cite]`。

**3. 推理时的图像一致性维护**

推理时，若生成器预测出 `[IMG]` token，则开始生成带引用标签的图像描述 $u_t'$。提取引用标签 $c_t$ 后，将文本描述 $u_t$ 与所有具有相同引用标签的历史图像一起送入 **BLIP-Diffusion** 进行定制化生成：

$$r_t^{\text{Image}} = \mathcal{F}(u_t \mid \{r_i^{\text{Image}} \mid c_i = c_t\}_{i=1}^{t-1})$$

若无历史引用匹配，则使用标准 **Stable Diffusion 2.1** 生成。

### 损失函数 / 训练策略

采用两阶段训练：

- **第一阶段**：仅训练语言模型层 $\theta_{\mathcal{G}_l}$，batch size=256，最大 token 长度=256
- **第二阶段**：联合训练 Visual Encoder 的 perceiver resampler $\theta_{\mathcal{V}}$ 和视觉交叉注意力层 $\theta_{\mathcal{G}_v}$，batch size=128，最大 token 长度=512

两阶段均使用标准的 next token prediction 损失（负对数似然），AdamW 优化器，学习率 1e-4，16 × NVIDIA A100 80GB 训练。

## 实验关键数据

### 主实验

在 PhotoChat 和 MMDialog 上的综合评测：

| 模型 | Intent F1 | IS | TID B1 | TID B2 | TID R-1 | TID R-L | TR B1 | TR B2 | TR R-1 | TR R-L |
|------|-----------|-----|--------|--------|---------|---------|-------|-------|--------|--------|
| **PhotoChat** | | | | | | | | | | |
| Divter | 56.2 | 15.8 | 15.1 | 11.4 | - | 15.8 | 6.52 | 1.66 | - | 5.69 |
| Divter_LLM (3B) | 54.1 | 16.1 | 41.3 | 27.1 | 43.3 | 41.6 | 11.4 | 4.75 | 11.2 | 10.8 |
| **BI-MDRG** | **55.7** | **16.7** | **42.1** | **28.2** | **44.6** | **42.5** | **12.4** | **5.12** | **12.1** | **11.2** |
| **MMDialog** | | | | | | | | | | |
| Divter | 71.8 | 20.5 | - | - | - | - | 9.44 | 7.45 | - | 11.2 |
| MiniGPT5 (9B) | - | 20.2 | - | - | - | - | 29.1 | 19.5 | - | 12.1 |
| Divter_LLM (3B) | 67.3 | 21.0 | 44.2 | 35.7 | 45.5 | 43.6 | 21.3 | 16.2 | 20.4 | 19.4 |
| **BI-MDRG (4B)** | **70.5** | **22.4** | **52.2** | **44.7** | **53.2** | **51.6** | **27.6** | **23.5** | **25.7** | **24.8** |

图像 grounding 评测（ImageChat，零样本迁移）：

| 模型 | B1 | R-1 | R-L |
|------|-----|------|------|
| Divter_LLM | 8.6 | 10.3 | 9.6 |
| BI-MDRG w/o mask | 10.0 | 11.1 | 10.2 |
| **BI-MDRG** | **10.9** | **11.7** | **10.9** |

### 消融实验

**Citation 框架对图像一致性的影响**（MDIC 数据集）：

| Citation 方式 | VLM 大小 | Diffusion | DINOv2 ↑ |
|--------------|---------|-----------|----------|
| **Citation Module** | 4B | Custom+Diffusion | **0.53** |
| LLMCite | 4B | Custom+Diffusion | 0.34 |
| 无 Citation | 4B | Diffusion | 0.25 |
| LLMCite | 9B | Custom+Diffusion | 0.33 |
| 无 Citation | 9B | Diffusion | 0.26 |

**Citation Token 预测准确率**：

| 模型 | Acc. | DINOv2 ↑ |
|------|------|----------|
| Divter_LLM + LLMCite | 33.5 | 0.32 |
| **BI-MDRG** | **84.0** | **0.53** |

**Citation Module 自身质量**：对 MDIC 数据集 300 条对话的伪标签质量评估，F1 = 0.72。

### 关键发现

1. **BI-MDRG (4B) 在 MMDialog 上全面超越 MiniGPT5 (9B)**，说明架构设计比单纯增大模型更有效
2. **注意力掩码调制有效**：在 ImageChat 上，BI-MDRG 相比无掩码版本 B1 从 10.0 提升到 10.9，R-L 从 10.2 到 10.9
3. **模型规模增大无法解决一致性问题**：4B→9B 提升了文本回复质量，但 DINOv2 一致性分数几乎不变（0.25→0.26），必须依赖 Citation 框架
4. **Citation Module 远优于 LLM 指令方式**：84.0% vs 33.5% 的引用预测准确率，说明视觉特征聚类比纯文本推理更可靠

## 亮点与洞察

1. **问题定义精准**：清晰地分析了"文本中间媒介"范式的两个根本缺陷（grounding 不足 + 一致性缺失），并针对性设计了两条桥接路径
2. **Citation Module 设计巧妙**：全部使用现成预训练模型组合（spaCy + GroundingDINO + SAM + DINOv2），零额外训练开销，模块化可替换
3. **注意力掩码调制**：简洁优雅的方法强制模型从视觉特征而非文本描述获取图像信息，实现成本极低
4. **创建了 MDIC 评测数据集**：填补了多模态对话图像一致性评测的空白，300 条手工标注对话
5. **揭示了一个重要洞察**：单纯增大模型无法解决图像一致性问题，必须有专门的框架来维持

## 局限性 / 可改进方向

1. **Pipeline 式架构**：依赖多个独立组件串联（POS tagger → 检测器 → 分割 → 特征提取 → 聚类），错误会逐级传播
2. **单物体追踪限制**：Citation Module 每轮只提取一个主要物体词，无法处理多物体同时需要一致性的场景
3. **MDIC 数据集规模小**：仅 300 条对话的评测集，统计意义有限
4. **依赖文本中间表示**：尽管增加了视觉桥接，图像生成仍需通过文本描述→text-to-image 的路径，本质上是在旧范式上做增量改进
5. **BLIP-Diffusion 的定制化能力有限**：零样本 subject-driven 生成质量受限于定制化模型本身

## 相关工作与启发

- **Flamingo**：BI-MDRG 的架构直接借鉴 Flamingo 的交叉注意力设计，将视觉特征注入语言模型
- **Divter**：BI-MDRG 的直接前驱工作，建立了 MDRG 任务的 text-as-intermediary 范式
- **BLIP-Diffusion**：提供零样本 subject-driven 图像生成能力，是实现图像一致性的底层技术
- **DINOv2 + SAM + GroundingDINO**：展示了 foundation model 的 pipeline 组合能力，无需训练即可完成复杂的物体追踪任务
- **启发**：在端到端多模态生成模型尚不成熟时，如何巧妙利用模块化组件和注意力机制来弥补 pipeline 方法的信息损失

## 评分

- 新颖性: ⭐⭐⭐⭐ — Citation Module 和注意力掩码调制设计新颖，但整体仍在已有范式上做增量
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集 + 自建 MDIC 评测集 + 多维度消融，较为完整
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，图示说明力强，结构合理
- 价值: ⭐⭐⭐⭐ — 多模态对话一致性是实际重要问题，方法实用且开源
