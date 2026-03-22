# Unleashing Video Language Models for Fine-grained HRCT Report Generation

**会议**: CVPR 2025  
**arXiv**: [2603.12469](https://arxiv.org/abs/2603.12469)  
**代码**: [GitHub](https://anonymous.4open.science/r/hrct-report-generation-video-vlm-728C/)  
**领域**: medical_imaging  
**关键词**: HRCT 报告生成, 视频语言模型, Chain-of-Thought, DPO, 异常检测, 3D 医学影像

## 一句话总结

提出 AbSteering 框架，通过异常中心化 CoT 训练和基于临床混淆异常硬负例的 DPO 优化，将通用视频语言模型（VideoLMs）高效迁移到 HRCT 报告生成任务，性能超越专用 CT 基础模型。

## 研究背景与动机

1. **HRCT 报告生成的临床价值**：HRCT 是胸部和心肺疾病的关键诊断模态，AI 自动报告生成可减少临床工作量、标准化诊断叙述、缓解观察者间差异
2. **从 X-ray 到 CT 的挑战**：相比 2D X-ray，HRCT 引入（1）数百张切片带来的计算和内存开销，（2）更难的视觉理解问题——临床关键异常细微、空间稀疏、多样
3. **现有方法的局限**：专用 CT 基础模型（RadFM、CT-CHAT、M3D）需要大规模 CT 数据预训练，且在细粒度长尾异常识别上仍不足
4. **VideoLMs 的潜力**：HRCT 体积可自然视为"视频式切片序列"，VideoLMs 具有强大时空推理能力，但缺乏医学领域知识
5. **核心问题**：（1）VideoLMs 的编码器能否捕获 3D 临床特征？（2）如何高效适配到医学报告？（3）与 CT 专用模型相比如何？
6. **核心 idea**：VideoLMs 和 CT 专用模型架构高度相似（3D tokenization + attention + LLM decoder），差异仅在训练域——因此关键在于高效的领域适配

## 方法详解

### 整体框架：AbSteering（两阶段）

利用预训练 VideoLM 作为 backbone，分两阶段进行领域适配：

### Stage 1: 异常中心化 CoT 训练

- **报告结构化**：将 CT-RATE 报告按 10 个解剖区域（肺、气管与支气管、纵隔、心脏、食管、胸膜、骨骼、甲状腺、乳腺、腹部）重组为统一的 (region: abnormality) 模板，使用 GPT-4o 辅助分类
- **CoT 训练**：训练目标序列 $Y = [R_{AB}; R_{Full}]$，模型先生成结构化异常发现（reasoning anchor），再合成完整报告
- **设计动机**：强制模型先进行异常推理，抑制正常组织主导的描述和幻觉，学习疾病共现/互斥等临床关联
- **数据集**：策划 CT-RATE-AB 数据集，包含结构化异常标注

### Stage 2: 细粒度异常判别（DPO）

- **硬负例构造**：使用 GPT-4o 将真实异常替换为同一解剖区域内的临床易混淆异常，生成 $R_{AB\_Fake}$，保持报告流畅性和结构一致性
- **DPO 优化**：$\mathcal{L}_{DPO}$ 让模型偏好真实报告 $R_{AB}$（winning）而非伪造报告 $R_{AB\_Fake}$（losing），迫使模型关注区分两者的细微视觉线索
- **设计动机**：CT 异常常表现为微妙且视觉易混淆的模式，DPO 通过对比学习增强异常判别能力并抑制幻觉

### 架构说明

VideoLMs 的视觉编码器将 CT 输入为 $X \in \mathbb{R}^{T \times H \times W \times C}$，通过时空 3D attention 编码，token merger 压缩后送入 LLM。与 CT 专用模型架构本质相同，差异在于预训练域。

## 实验关键数据

### 数据集
- CT-RATE：25,692 非增强胸部 CT（21,304 患者），扩展至 50,188 volumes，每个 CT 转为 240 帧 480×480 的 MP4（18fps）
- 训练集 46,717 CT（20,000 患者），验证集 3,039 CT（1,314 患者）

### 主实验（CT-RATE benchmark）

| 方法 | BL-1 | RG-L | BERT | CE Micro F1 | CE Macro F1 |
|------|------|------|------|-------------|-------------|
| M3D-8B | 44.95 | 37.76 | 87.52 | 35.69 | 26.74 |
| Qwen2.5-VL-7B | 43.67 | 36.71 | 87.30 | 33.64 | 25.57 |
| InternVL3-8B | 45.57 | 38.49 | 87.40 | 44.45 | 38.91 |
| M3D-AbSteer | 45.22 | 38.58 | 87.83 | 43.24 | 36.18 |
| Qwen-AbSteer | 45.64 | 37.99 | 87.13 | 45.99 | 37.90 |
| **InternVL3-AbSteer** | **48.32** | **40.49** | **87.59** | **54.55** | **47.66** |

### 关键发现
- 通用 VideoLMs（Qwen2.5-VL、InternVL3）fine-tune 后即可匹配 CT 专用基础模型 M3D
- AbSteering 对 VideoLMs 的提升远大于对 M3D 的提升（InternVL3 CE Micro F1: 44.45→54.55）
- **InternVL3-AbSteer 在所有临床效能指标上大幅超越所有 CT 专用模型**

### 消融实验
- CoT 显著提升 recall，DPO 在 CoT 基础上同时提升 precision 和 recall
- 视频预训练至关重要：从头训练性能大幅下降；LoRA 无增益（冻结编码器即可）
- LLM 规模：3B→7B 提升，7B→32B 反而下降，瓶颈在视觉-文本对齐而非 LLM 容量

## 亮点

- **跨模态迁移的新范式**：证明通用 VideoLMs 在有限数据下可高效迁移到 3D 医学影像，无需从头训练专用基础模型
- **CoT + DPO 的协同**：CoT 提升异常召回率，DPO 抑制幻觉——两者通常难以同时优化
- **临床混淆硬负例**：用同区域易混淆异常构造 DPO 负例，精准针对细粒度判别瓶颈
- **"冻结编码器即可"的发现**：VideoLM 预训练特征足够鲁棒，无需额外适配视觉编码器

## 局限性

- 仅在 CT-RATE 单一数据集验证，缺乏跨机构/跨疾病谱的泛化性评估
- GPT-4o 依赖较重（报告结构化 + 硬负例生成），增加了数据准备的成本和可复现性风险
- 仅针对胸部 HRCT，未验证对腹部/头部 CT 等其他部位的适用性
- LLM 规模 32B 反而下降的现象值得进一步探究（数据密度不足 vs 过拟合）
- 将 CT 转为 MP4 视频格式（240帧 18fps）可能引入有损压缩伪影，对细微异常识别的影响未讨论
- NLG 指标提升有限（BLEU-4 仅 23.58），临床效能指标提升才是核心贡献

## 与相关工作的对比

- **vs CT-CHAT/RadFM（CT 专用基础模型）**：架构相似但预训练域不同。AbSteering 证明通用视频预训练 + 高效适配 > 领域专用大规模预训练，且训练成本更低
- **vs M3D-8B（最强 CT 基础模型）**：M3D-AbSteer 的临床效能提升不如 VideoLM-AbSteer 显著，说明 VideoLMs 的通用时空推理能力更具可塑性
- **vs 传统 CoT 方法**：本文 CoT 不是通用推理链，而是领域特化的"异常发现→报告生成"因果链，与 DPO 协同更紧密
- **vs Dia-LLaMA**：Dia-LLaMA 设计 CT 专用视觉编码器对接 LLM，本文证明直接复用 VideoLM 的编码器即可，无需领域特化编码器

## 评分
- 新颖性: ⭐⭐⭐⭐ VideoLM→HRCT 迁移路径新颖，CoT+DPO 组合巧妙
- 实验充分度: ⭐⭐⭐⭐ 多 backbone 对比、消融全面、case study 充分
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，架构等价性分析有说服力
- 价值: ⭐⭐⭐⭐ 为 3D 医学报告生成提供了高效实用的新范式
- 总评: 8/10
