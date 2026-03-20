# The Coherence Trap: MLLM-Crafted Narratives Exploit Manipulated Visual Contexts

**会议**: CVPR 2026  
**arXiv**: [2505.17476](https://arxiv.org/abs/2505.17476)  
**代码**: [https://github.com/YcZhangSing/AMD](https://github.com/YcZhangSing/AMD)  
**领域**: AI安全 / 深度伪造检测  
**关键词**: multimodal manipulation detection, MLLM-driven disinformation, semantic alignment, deepfake grounding, dataset  

## 一句话总结
揭示现有多模态虚假信息检测的两个根本缺陷（低估MLLM生成的语义一致虚假叙事+依赖简单不对齐的伪影），构建441k样本的MDSM数据集（图像篡改+MLLM生成语义对齐文本），并提出AMD框架（Artifact Pre-perception + Manipulation-Oriented Reasoning），在跨域检测中达88.18 ACC / 60.25 mAP / 61.02 mIoU。

## 背景与动机
多模态假新闻检测面临新挑战：(1) 现有方法（DGM4/HAMMER）主要应对规则化文本篡改（如简单替换人名），忽视了MLLM能根据篡改图像动态生成流畅、上下文合理但误导性的叙事——这种"语义一致性陷阱"使传统对比学习失效；(2) 现有数据集中图-文篡改独立进行，产生的语义不一致很容易被公众识别。真实场景中的攻击者会刻意维护视觉-文本一致性以最大化误导效果。

## 核心问题
如何检测和定位语义一致的MLLM驱动多模态篡改——即图像被编辑后文本由MLLM重新生成以保持视觉-文本匹配？

## 方法详解

### MDSM数据集构建
- **数据源**：GoodNews/VisualNews/N24News，210万+图文对，过滤出含人脸+人名的数据
- **图像篡改**：Face Swap（SimSwap/e4s）和Face Attribute（StyleCLIP/HFGI，反转情绪）
- **文本篡改**：用Qwen2-VL生成语义对齐的虚假叙事——输入篡改后图像+人名列表（元信息），MLLM生成与视觉一致但虚假的文本
- **5种篡改组合**：FS / FS&TF / FA / FA&TF / TF
- **规模**：441k样本，来自Guardian/NYT/USA Today/Washington Post/BBC，支持跨域评估

### AMD框架
基于Florence-2，三阶段pipeline：

1. **Multi-modal Input Embedding**：可学习Artifact Token $E_a \in \mathbb{R}^{n_a \times d}$ 拼接到图像和文本嵌入之间 $S_{inp} = [E_v; E_a; E_t]$

2. **Artifact Pre-perception Encoding (APE)**：通过冻结的预感知编码器 $\mathcal{E}_{mp}$ 处理输入序列，提取artifact token $\hat{E}_a$，用二分类头（加权池化+分类器）做篡改检测监督 $\mathcal{L}_{APE}$，将伪影线索注入artifact token。关键：冻结编码器参数以保存MLLM原始知识，替换回原始图-文嵌入以保持推理能力。

3. **Manipulation-Oriented Reasoning (MOR)**：
   - **Visual Artifact Aggregation (VAA)**：artifact token作query通过交叉注意力从视觉特征中聚合篡改空间信息，用于bbox定位（$\mathcal{L}_{IMG}$）
   - **Dual-Branch Manipulation (DBM)**：视觉+artifact和文本分别作query做跨模态交叉注意力，双分支二分类判别（$\mathcal{L}_{DBM}$）——mAP从47.18提升到66.47
   - **Language Modeling**：自回归生成文本答案（选项+坐标）

4. **Token Redundancy Penalty (TRP)**：正交约束 $\mathcal{L}_{orth}$ + KL散度均匀约束 $\mathcal{L}_{mod}$，防止artifact token冗余坍缩

### 总损失
$\mathcal{L} = \mathcal{L}_{APE} + \mathcal{L}_{DBM} + \mathcal{L}_{IMG} + \mathcal{L}_{TRP} + \mathcal{L}_{LM}$

推理时所有辅助头丢弃，仅保留文本生成。

## 实验关键数据

| MDSM (Guardian训练) | 方法 | AVG ACC↑ | AVG mAP↑ | AVG mIoU↑ |
|---|---|---|---|---|
| | ViLT | 76.61 | 49.90 | 35.67 |
| | HAMMER++ | 75.10 | 49.01 | 48.49 |
| | FKA-Owl (7B) | 84.12 | 58.13 | 52.20 |
| | **AMD (0.27B)** | **88.18** | **60.25** | **61.02** |

| DGM4 (Guardian训练) | AVG ACC↑ | AVG mAP↑ | AVG mIoU↑ |
|---|---|---|---|
| HAMMER++ | 65.61 | 47.36 | 46.19 |
| FKA-Owl | 71.96 | 42.68 | 44.15 |
| **AMD** | **74.47** | **52.91** | **51.87** |

- Zero-shot通用模型（GPT-4o/Gemini-2.0/Qwen3-VL-235B）在MDSM上仅~33% ACC——语义一致篡改对当前MLLM极具挑战
- 跨MLLM泛化：用Qwen-VL/LLaVA/mPLUG-Owl/X-InstructBLIP生成文本测试，AMD保持53+ AP
- 效率：276M参数 vs FKA-Owl 6771M，推理13.38 p/s vs FKA-Owl 1.33 p/s

### 消融实验要点
- APE: ACC 76.92→82.93（+6）——预感知伪影线索是关键
- DBM: mAP 47.18→66.47（+19）——双分支跨模态判别大幅提升分类
- IMG: mIoU 60.13→61.78——grounding辅助任务有帮助
- TRP: ACC和mIoU全面小幅提升——减少token冗余
- t-SNE可视化：Artifact token经三阶段处理后类别聚类逐渐清晰

## 亮点
- 定义了极具现实意义的新问题：MLLM驱动的语义一致多模态篡改——比规则化文本替换更难检测
- MDSM数据集填补关键空白：441k规模、5个媒体域、语义对齐、支持跨域评估
- AMD仅276M参数却超越7B级FKA-Owl——统一的seq2seq框架比多头架构更高效
- APE的"冻结编码器+仅训练artifact token"策略巧妙保留MLLM知识
- 伦理考虑周全：不发布生成pipeline/prompt，仅限研究访问，图像加水印

## 局限性 / 可改进方向
- 仅考虑人脸相关篡改（换脸/属性编辑），其他物体或场景级篡改未覆盖
- 文本篡改仅用Qwen2-VL生成——虽然验证了跨MLLM泛化，但未覆盖最新的推理型模型
- Florence-2作为backbone规模较小，更大backbone可能进一步提升
- 跨域泛化在部分域上仍有明显gap（如NYT训练→USA测试时性能下降）

## 与相关工作的对比
- **vs DGM4/HAMMER**：DGM4独立篡改图文导致语义不一致容易检测；MDSM对齐篡改更难，HAMMER在MDSM上mAP仅44
- **vs MMFakeBench**：仅30%语义对齐样本且11k规模不足训练；MDSM 100%对齐、441k规模
- **vs FKA-Owl**：7B模型、仅做二分类无细粒度分类/定位；AMD 0.27B、统一检测+分类+定位
- **vs 通用MLLM（GPT-4o等）**：零样本~33% ACC——说明需要专门训练检测语义一致篡改

## 启发与关联
- MLLM生成的虚假信息是真实存在的社会安全威胁——该工作为防御此类攻击提供了基础设施
- Artifact token的pre-perception设计可推广到其他需要检测特定信号的MLLM应用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性定义和应对MLLM驱动的语义一致多模态篡改问题
- 实验充分度: ⭐⭐⭐⭐⭐ 441k数据集、5域跨域、4种MLLM泛化、零样本通用模型对比、完整消融
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰、数据集构建详尽、伦理考虑完善
- 价值: ⭐⭐⭐⭐⭐ 数据集+方法共同定义了MLLM时代虚假信息检测的新范式
