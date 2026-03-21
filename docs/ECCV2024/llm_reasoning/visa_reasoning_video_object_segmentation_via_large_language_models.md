# VISA: Reasoning Video Object Segmentation via Large Language Models

**会议**: ECCV 2024  
**arXiv**: [2407.11325](https://arxiv.org/abs/2407.11325)  
**代码**: [https://github.com/cilinyan/VISA](https://github.com/cilinyan/VISA) (有)  
**领域**: LLM推理  
**关键词**: Reasoning Segmentation, Video Object Segmentation, Multi-Modal LLM, SAM, 视频推理

## 一句话总结

提出 ReasonVOS 新任务和 VISA 模型，利用多模态 LLM 的世界知识推理能力实现基于隐式文本查询的视频目标分割与跟踪。

## 研究背景与动机

1. **领域现状**：现有视频目标分割（VOS）依赖显式用户指令——类别、掩码或短语（如"正在跑的车"），能力受限于直觉可见的特征描述。
2. **现有痛点**：用户的真实需求往往是隐式的（如"找到我最喜欢的杯子"），需要模型具备常识推理和视频内容理解能力，而现有方法完全不具备这种能力。
3. **核心矛盾**：推理分割（Reasoning Segmentation）在图像领域已有 LISA 等工作，但视频领域需要同时处理时序信息和空间细节，每帧需大量 visual token，直接扩展到视频 domain 计算上不可行。
4. **本文要解决什么**：如何在视频中实现需要世界知识推理的目标分割——给定隐式、复杂的文本指令（如"电动汽车""哪辆车最可能赢比赛"），输出对应目标的掩码序列。
5. **切入角度**：不全帧处理，而是用 Text-guided Frame Sampler 先选出关键帧，减少 token 数量；然后用多模态 LLM 同时处理多帧做推理，配合 SAM decoder 分割 + tracker 传播。
6. **核心 idea 一句话**：通过文本引导的关键帧采样 + 多模态 LLM 推理 + SAM 解码 + 目标跟踪的流水线，高效实现视频推理分割。

## 方法详解

### 整体框架

VISA 由三个核心组件组成，构成一个优雅的流水线：

1. **Text-guided Frame Sampler (TFS)**：输入完整视频 + 文本描述 → 输出一帧目标帧 $f_{tgt}$ 和 $T_r$ 帧参考帧 $\mathbf{x}_r$
2. **Multi-Modal LLM**：将目标帧、参考帧和文本 token 拼接输入 LLM → 输出包含特殊 `<Seg>` token 的文本 → 提取 `<Seg>` 的 hidden embedding 经 MLP 投影为 SAM 的 prompt embedding → SAM decoder 解码目标帧的分割掩码 $m_{tgt}$
3. **Object Tracker (XMem)**：将 $m_{tgt}$ 双向传播到所有剩余帧 → 得到完整掩码序列 $\mathcal{M}$

### 关键设计

**1. Text-guided Frame Sampler (TFS) —— 文本引导关键帧采样**

- **做什么**：从长视频中选出与文本查询最相关的目标帧和参考帧。
- **核心思路**：利用 LLaMA-VID（能把每帧压缩为 2 个 token 处理长视频）作为采样器。设计 prompt 模板："To find {description}, which percentage mark of the video should I check?"，让 LLaMA-VID 生成 Top-K=10 个响应，取百分比平均值定位目标帧 $f_{tgt}$。
- **设计动机**：视频中大部分帧与查询无关（如问"谁会接棒"只需看最后几帧），直接处理全部帧 $T \times L$ 个 token 计算不可行。TFS 既保留空间细节（不做 spatial pooling），又大幅减少需要处理的帧数。
- **参考帧采样策略**：基于 $f_{tgt}$ 采样 $T_r$ 帧参考帧，支持 Global（全视频均匀采样）、Local（以 $f_{tgt}$ 为中心的连续采样）和 Global-Local（各取一半组合）三种策略。

**2. Multi-Modal LLM + `<Seg>` Token 机制 —— 推理与分割桥接**

- **做什么**：对选中的帧做视觉-语言联合推理，并输出可解码为分割掩码的嵌入。
- **核心思路**：采用 Chat-UniVi（支持灵活 visual token 数的多模态 LLM）作为骨干。每帧经 ViT 编码后通过 Spatial Merging 得到 $L=112$ 个 visual token。目标帧、参考帧和文本 token 拼接后输入 LLM，生成包含 `<Seg>` 的回答。提取 `<Seg>` 最后一层 embedding，经 MLP 投影为 $h_{seg}$，作为 SAM decoder 的 prompt embedding。
- **设计动机**：沿用 LISA 的 `<Seg>` token 设计，将语言模型的推理能力与分割能力解耦——LLM 负责推理"是什么"，SAM decoder 负责精确分割"在哪里"。同时处理多帧可获取时序信息，优于 TrackGPT 的逐帧处理。

**3. Object Tracker (XMem) —— 双向掩码传播**

- **做什么**：将目标帧的分割掩码传播到视频所有帧。
- **核心思路**：使用冻结的 XMem 做双向传播：$\mathcal{M} = \text{OT}(m_{tgt}, \mathbf{x}_v)$
- **设计动机**：只需对一帧做精确分割，利用成熟的 VOS tracker 完成全视频的时间一致性分割，避免对每帧都做推理的高计算开销。

### 损失函数 / 训练策略

**损失函数**：$\mathcal{L} = \lambda_{txt} \mathcal{L}_{txt} + \lambda_{mask} \mathcal{L}_{mask}$

- $\mathcal{L}_{txt}$：自回归交叉熵损失（文本生成）
- $\mathcal{L}_{mask} = \lambda_{bce} \text{BCE}(\hat{m}, m) + \lambda_{dice} \text{DICE}(\hat{m}, m)$
- 权重设置：$\lambda_{txt}=1.0$，$\lambda_{mask}=1.0$，$\lambda_{bce}=2.0$，$\lambda_{dice}=0.5$

**训练策略**：

- 冻结模块：TFS、Vision Backbone、Object Tracker 全部冻结
- 可训练模块：Multi-Modal LLM（LoRA 高效微调）+ SAM Decoder
- 训练时随机采样目标帧和 8-12 帧参考帧（不用 TFS），推理时才用 TFS 选帧
- 训练数据混合：Referring VOS（Ref-YouTube-VOS, MeViS, Ref-DAVIS17）+ Video QA + Image 数据 + ReVOS
- 硬件：8×A100 80G，batch size 128，10 epochs，AdamW + cosine schedule，lr=2e-5

## 实验关键数据

### 主实验

**Table 1: ReVOS 数据集性能对比**

| 方法 | Backbone | Referring J&F | Reasoning J&F | Overall J&F | R (鲁棒性) |
|------|----------|:---:|:---:|:---:|:---:|
| ReferFormer | Video-Swin-B | 32.7 | 23.4 | 28.1 | 8.8 |
| LISA | LLaVA-7B | 45.7 | 36.1 | 40.9 | 9.3 |
| TrackGPT(IT) | LLaVA-7B | 48.2 | 39.0 | 43.6 | 11.6 |
| TrackGPT(IT) | LLaVA-13B | 49.5 | 40.5 | 45.0 | 12.8 |
| **VISA(IT)** | Chat-UniVi-7B | 50.9 | 43.0 | 46.9 | 15.5 |
| **VISA(IT)** | Chat-UniVi-13B | **57.4** | **44.3** | **50.9** | 14.5 |

**Table 2: Referring VOS 数据集性能对比**

| 方法 | Backbone | MeViS J&F | Ref-YT-VOS J&F | Ref-DAVIS17 J&F |
|------|----------|:---:|:---:|:---:|
| ReferFormer | Video-Swin-B | 31.0 | 62.9 | 61.1 |
| LISA | LLaVA-13B | 37.9 | 54.4 | 66.0 |
| TrackGPT | LLaVA-13B | 41.2 | 59.5 | 66.5 |
| **VISA** | Chat-UniVi-7B | 43.5 | 61.5 | 69.4 |
| **VISA** | Chat-UniVi-13B | **44.5** | **63.0** | **70.4** |

### 消融实验

**Table 5: 目标帧选择与参考帧采样策略消融（ReVOS Overall J&F）**

| 目标帧 | $T_r$ | 无采样 | Global | Local | Global-Local |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $f_0$（首帧） | 0 | 42.6 | - | - | - |
| $f_0$ | 12 | - | 44.5 | 44.9 | 45.0 |
| $f_{tgt}$（TFS） | 0 | 44.3 | - | - | - |
| $f_{tgt}$ | 12 | - | 46.7 | 46.3 | **46.9** |

- 使用 TFS 选帧比直接用首帧提升 ~2%
- Global-Local 组合策略最优
- 12 帧参考帧优于 6 帧

**Table 4: 训练数据集消融（ReVOS, Chat-UniVi-7B）**

| ReferVOS | VQA | Image | ReVOS | Referring J&F | Reasoning J&F |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✓ | ✓ | ✓ | 47.6 | 39.9 |
| ✓ | ✓ | ✗ | ✓ | 34.2 | 33.5 |
| ✓ | ✓ | ✓ | ✗ | 52.9 | 39.2 |
| ✓ | ✓ | ✓ | ✓ | 50.9 | **43.0** |

### 关键发现

- 传统方法（如 ReferFormer）在 reasoning 任务上 J&F 仅 ~23%，VISA(IT) 达到 44.3%，提升超过 20 个点
- VISA 多帧处理优于 TrackGPT 的逐帧处理，overall J&F 高 3.3%（7B 模型）
- Image 数据集对训练至关重要——去掉后性能暴跌 16.7%（referring）/ 9.5%（reasoning），因为图像数据规模远大于视频数据
- ReVOS 训练集的指令微调让 reasoning 提升 3.8%，同时 referring 性能几乎不变
- 每帧 visual token 数 L=112 和 L=256 性能相当，但 L=56 显著下降
- 包含不存在目标的负样本训练让鲁棒性分数 R 提升到 15.5（vs TrackGPT 11.6）

## 亮点与洞察

- **任务定义前瞻性**：ReasonVOS 是一个很好的新任务定义——将 VOS 从显式指令扩展到需要世界知识推理的隐式指令，更贴近真实的 Embodied AI 需求
- **架构设计精巧解耦**：TFS 选帧 → LLM 推理 → SAM 分割 → Tracker 传播，四个模块各司其职，冻住不需训练的模块（TFS、Backbone、Tracker），只训练 LLM（LoRA）和 SAM decoder，训练极其高效
- **`<Seg>` token 桥接机制**：继承 LISA 的设计，通过一个特殊 token 将 LLM 的推理能力无缝桥接到分割能力，避免了分割模型本身需要推理的问题
- **数据集贡献扎实**：ReVOS 包含隐式推理描述（14,678）、显式描述（20,071）和不存在目标描述（325），覆盖全面，且负样本的引入有效降低幻觉
- **训练时替代 TFS 的策略**：训练时随机采帧而非使用 TFS，让模型见到更多帧的组合，推理时再用 TFS 精确选帧——这个 train/test 策略差异设计得很合理

## 局限性 / 可改进方向

- **小目标问题**：每帧仅 112 个 visual token，小目标（如桨）分割效果差，增加 token 数会带来计算负担
- **关键帧定位依赖强**：TFS 定位不准会直接导致分割失败，例如仅出现在单帧的目标（如拿灭火器的人）很难被定位到
- **长时序建模受限**：虽采多帧参考，但 12 帧仍有限，对需要极长时序理解的查询力不从心
- **可替换的 backbone**：当前基于 Chat-UniVi，后续可替换为更强的多模态 LLM（如 InternVL、Qwen-VL）以持续提升
- **Tracker 误差累积**：XMem 的双向传播可能在长视频或复杂场景中出现漂移，分割质量随传播距离下降

## 相关工作与启发

- **vs LISA**：VISA 是 LISA 在视频域的自然扩展，核心差异在于加入了 TFS 和 Tracker 处理时序问题；VISA 在图像任务上也达到了与 LISA 相当的水平
- **vs TrackGPT**：TrackGPT 逐帧处理缺乏时序信息，VISA 同时处理多帧做推理，overall J&F 高 5.9（13B 模型）
- **vs LLaMA-VID + LMPM**：VQA+ReferVOS pipeline 的问题在于 VQA 模型的低 token 数不足以精确定位目标，两阶段误差累积
- **vs Video-ChatGPT**：通过 spatial/temporal pooling 压缩 token 的方法丢失了分割所需的空间细节，VISA 通过 TFS 选帧保留了空间分辨率

## 评分

- ⭐⭐⭐⭐ **新颖性**：ReasonVOS 新任务定义有价值，TFS + LLM + SAM + Tracker 的流水线设计简洁有效，但各组件均为已有模块的组合
- ⭐⭐⭐⭐ **实验充分度**：8 个数据集、多组消融实验（数据集/帧采样/token数/采样策略），横向纵向对比都很充分
- ⭐⭐⭐⭐ **写作质量**：结构清晰，任务定义明确，图表丰富，动机阐述到位
- ⭐⭐⭐⭐⭐ **价值**：开创了 Reasoning VOS 方向，ReVOS 数据集填补空白，代码和数据集开源，对后续研究有较大推动
