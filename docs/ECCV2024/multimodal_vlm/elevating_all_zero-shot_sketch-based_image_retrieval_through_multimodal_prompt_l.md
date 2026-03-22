# SpLIP: 通过多模态提示学习提升所有零样本草图检索任务

**会议**: ECCV2024  
**arXiv**: [2407.04207](https://arxiv.org/abs/2407.04207)  
**代码**: [Project Page](https://mainaksingha01.github.io/SpLIP/)  
**领域**: multimodal_vlm  
**关键词**: CLIP, 草图检索(SBIR), 多模态提示学习, 零样本学习, 跨模态对齐

## 一句话总结

提出 SpLIP，一种基于冻结 CLIP 的双向多模态提示学习框架，通过视觉-文本编码器间的双向知识交换、自适应 margin 的三元组损失和条件跨模态拼图任务，在 ZS-SBIR、GZS-SBIR 和 FG-ZS-SBIR 三种草图检索设定下均取得 SOTA。

## 研究背景与动机

### 任务定义

草图检索（Sketch-Based Image Retrieval, SBIR）旨在根据手绘草图从图库中检索对应的照片。核心挑战在于草图与照片之间存在巨大的域差异（domain gap）。该任务有三种设定：

- **ZS-SBIR**：测试类别在训练时完全未见
- **GZS-SBIR**：推理时图库同时包含训练类和未见类照片，模型容易偏向训练类
- **FG-ZS-SBIR**：在实例级别进行精细匹配，而非类别级别

### 现有方法的不足

1. **单模态提示学习**：现有基于 CLIP 的 SBIR 方法（如 CLIP-AT、TLT）主要使用单模态（视觉）提示，未充分利用 CLIP 的视觉-文本双路径协同能力
2. **多模态提示的局限**：已有多模态提示方法（MaPLe、PromptSRC）仅支持单向 token 共享，文本路径对视觉信息不敏感，语义深度受限
3. **拼图策略粗糙**：CLIP-AT 的 patch shuffling 策略对正负对使用相同/不同排列进行三元组学习，无法有效建立局部与全局的对应关系

### 本文动机

需要一种**双向**的跨模态知识交换机制，在 CLIP 的视觉和文本编码器之间建立更紧密的协同，同时引入条件性的跨模态拼图任务来增强细粒度对齐。

## 方法详解

### 整体框架

SpLIP 基于冻结的 CLIP（ViT-B/32），包含以下核心组件：

1. **视觉引导的深度文本提示**（Vision-guided Textual Prompting）
2. **文本引导的深度视觉提示**（Text-guided Visual Prompting）
3. **条件跨模态拼图求解器**（Conditional Cross-modal Jigsaw Solver）
4. **自适应 margin 三元组损失**

训练时只优化 LayerNorm 参数和三个映射模块（$\mathcal{B}_t, \mathcal{B}_v, \mathcal{B}_{vt}$）以及拼图求解器 $\mathcal{F}_{js}$，CLIP 主干完全冻结。

### 关键设计 1：双向提示共享

**视觉→文本方向**：通过映射块 $\mathcal{B}_t$ 将图像 patch 嵌入 $\mathbf{E}_0$ 转换为 $m=4$ 个可学习的文本 token $\mathbf{T}$，注入到文本编码器 $\mathcal{F}_t$ 的每一层。与 MaPLe 等方法的随机初始化不同，这里的文本 prompt 直接捕获了视觉分布。

**文本→视觉方向（双通道）**：
- **通道一**：通过 $\mathcal{B}_v$ 将文本 token（"sketch/photo of a"，不含 [CLS]）映射为 $\mathcal{J}-1$ 个视觉 prompt token $\mathbf{V}^{\text{tg}}$，每层相同
- **通道二**：通过 $\mathcal{B}_{vt}$ 将文本编码器每层输出 $\mathbf{W}_l$（包含**所有训练类**的 token）转换为 $n=2$ 个**逐层不同**的视觉 prompt $\mathbf{V}^{\text{ms}}$

关键创新点：$\mathbf{V}^{\text{ms}}$ 聚合了所有训练类的文本信息（类别无关），这种跨类知识共享有效减小了嵌入空间中的语义间隙。

### 关键设计 2：条件跨模态拼图任务

给定 anchor 草图 $s_a$ 及其排列版本 $s_a'$，正照片 $p_a^+$（同类/同实例）和负照片 $p_a^-$：

- 构建融合特征：$r = [\mathcal{F}_v(s_a), \mathcal{F}_v(s_a')]$，$r^+ = [\mathcal{F}_v(p_a^+), \mathcal{F}_v(s_a')]$，$r^- = [\mathcal{F}_v(p_a^-), \mathcal{F}_v(s_a')]$
- 拼图求解器 $\mathcal{F}_{js}$（2 层 Transformer encoder + classifier）需要预测排列索引
- 核心思想：正照片应比负照片更能帮助解决草图的拼图任务，因为正照片与草图具有相同的空间结构

与 CLIP-AT 的区别：CLIP-AT 让正对共享相同排列、负对使用不同排列；SpLIP 让排列后的草图与**未排列的**正照片配对，更有效地学习局部-全局对应关系。

### 关键设计 3：自适应 Margin 三元组损失

传统三元组损失使用固定 margin，SpLIP 利用 CLIP 文本编码器的类名嵌入动态计算 margin：

$$\mu(c^+, c^-) = \cos(\mathcal{F}_t(c^+), \mathcal{F}_t(c^-))$$

当正负类语义相近（cosine 相似度高）时，margin 更大，迫使模型更努力地区分相似类别。

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{triplet} + \alpha \cdot \mathcal{L}_{class} + \beta \cdot \mathcal{L}_{cjs}$$

- $\mathcal{L}_{triplet}$：自适应 margin 的跨模态三元组损失
- $\mathcal{L}_{class}$：基于 CLIP 文本 prompt 的图文分类损失（交叉熵）
- $\mathcal{L}_{cjs}$：条件拼图损失 = 拼图交叉熵 + hinge loss（约束正对比负对有更低的拼图 CE）

## 实验关键数据

### 主实验：类别级 ZS-SBIR（Table 1）

| 方法 | 骨干 | Sketchy-1 mAP | Sketchy-2 mAP@200 | TU-Berlin mAP | QuickDraw mAP |
|------|------|:---:|:---:|:---:|:---:|
| BDA | CNN | 43.7 | 55.6 | 37.4 | 15.4 |
| PSKD | ViT | 68.8 | 56.0 | 50.2 | 15.0 |
| ZSE-Ret | ViT | 73.6 | 50.4 | 56.9 | 14.2 |
| CLIP-AT | CLIP | - | 72.3 | 65.1 | 20.2 |
| TLT | CLIP | 77.9 | 66.1 | 61.5 | 27.8 |
| MARL | CLIP | - | 69.1 | 70.5 | 32.7 |
| **SpLIP** | **CLIP** | **80.2** | **76.4** | **73.1** | **34.2** |

### GZS-SBIR 结果（Table 2）

| 方法 | Sketchy-2 mAP@200 | Sketchy-2 P@200 | TU-Berlin mAP | TU-Berlin P@100 |
|------|:---:|:---:|:---:|:---:|
| STL (AAAI'23) | 63.4 | 53.8 | 40.2 | 49.8 |
| CLIP-AT | 55.6 | 62.7 | 60.9 | 63.8 |
| MARL | 62.3 | 68.5 | 62.6 | 67.8 |
| **SpLIP** | **68.2** | **74.5** | **66.7** | **70.3** |

SpLIP 在 GZS-SBIR 上超过次优方法 **+4.8%**（Sketchy）和 **+4.1%**（TU-Berlin）mAP。

### FG-ZS-SBIR 结果（Table 3，Sketchy 数据集）

| 方法 | Acc@1 | Acc@5 |
|------|:---:|:---:|
| CLIP-AT | 28.68 | 62.34 |
| MARL | 29.96 | 58.53 |
| **SpLIP** | **33.45** | **66.71** |

SpLIP 在 Top-1 上超过 CLIP-AT 近 **+5%**，超过 MARL **+3.5%**。

### 跨数据集泛化（Table 4，训练 Sketchy-Ext，测试其他）

| 方法 | TU-Berlin mAP | TU-Berlin P@100 | QuickDraw mAP | QuickDraw P@100 |
|------|:---:|:---:|:---:|:---:|
| CLIP-AT | 56.4 | 63.1 | 30.7 | 45.0 |
| **SpLIP** | **70.6** | **76.0** | **45.8** | **58.6** |

跨数据集场景下提升巨大：TU-Berlin **+14.2%** mAP，QuickDraw **+15.1%** mAP。

### 消融实验：损失函数（Table 5，Sketchy）

| 损失组合 | ZS mAP@200 | ZS P@200 | FG Acc@1 | FG Acc@5 |
|---------|:---:|:---:|:---:|:---:|
| $\mathcal{L}_{class}$ | 57.5 | 58.1 | 17.23 | 39.41 |
| $\mathcal{L}_{triplet}$ | 71.6 | 72.7 | 26.54 | 59.92 |
| $\mathcal{L}_{class} + \mathcal{L}_{triplet}$ | 73.1 | 73.9 | 30.07 | 62.95 |
| + $\mathcal{L}_{margin}$ | 74.5 | 75.1 | 31.23 | 63.54 |
| + $\mathcal{L}_{cjs}$（完整） | **76.4** | **77.3** | **33.45** | **66.71** |

### 消融实验：可学习模块（Table 6，Sketchy）

| 配置 | ZS mAP@200 | FG Acc@1 |
|------|:---:|:---:|
| w/o LayerNorm | 74.2 | 31.24 |
| w/o $\mathcal{B}_v$（无文本→视觉 prompt） | 71.9 | 29.76 |
| w/o $\mathcal{B}_t$（无视觉→文本 prompt） | 72.8 | 30.41 |
| w/o $\mathcal{B}_{vt}$（无跨层文本→视觉） | 70.5 | 28.92 |
| w/o 全部视觉 prompt | 68.8 | 26.54 |
| w/o 所有 prompt 模块 | 62.5 | 28.49 |
| **SpLIP (完整)** | **76.4** | **33.45** |

### 关键发现

1. **双向 prompt 共享至关重要**：移除任一方向的 prompt 共享都导致 ZS-SBIR 下降 4%+，FG-SBIR 下降 3%+
2. **$\mathcal{B}_{vt}$ 是最关键的模块**：移除后 mAP 下降 5.9%，因为它负责聚合所有训练类文本信息到视觉端
3. **自适应 margin 优于固定 margin**：动态 $\mu$ 在所有任务上显著优于固定 $\mu=0.2$
4. **条件拼图优于 CLIP-AT 的拼图策略**：替换为 CLIP-AT 的方案后性能明显下降
5. **深度 prompt（逐层注入）优于浅层 prompt**：随着参与的层数增加，mAP 持续提升
6. **跨数据集泛化能力极强**：在未见数据集上提升 14-15% mAP

## 亮点与洞察

1. **首个将多模态 prompt learning 应用于 SBIR 的工作**：之前的 CLIP-SBIR 方法仅使用视觉 prompt 或简单微调
2. **双向信息流设计精巧**：视觉→文本（通过 $\mathcal{B}_t$）让文本 prompt 感知图像内容；文本→视觉（通过 $\mathcal{B}_v + \mathcal{B}_{vt}$）让视觉编码器融入语义知识，形成闭环
3. **类别无关的知识聚合**：$\mathcal{B}_{vt}$ 聚合所有训练类的文本特征而非仅当前类，推理时无需见过测试类名称，天然支持零样本
4. **条件拼图任务的巧妙改进**：将"排列后草图 vs 排列后照片"改为"排列后草图 vs 完整照片"，更好地建立局部-全局对应
5. **自适应 margin 利用了 CLIP 的语义结构**：语义相近的类别需要更大的区分力度，这与人类直觉一致

## 局限性

1. **仅使用 ViT-B/32**：未探索更大的 CLIP 骨干（ViT-L/14），可能有进一步提升空间
2. **训练开销**：虽然 CLIP 冻结，但双向 prompt 共享和拼图求解器引入了额外计算（多个映射模块 + Transformer decoder）
3. **拼图任务的超参数**：排列数量 $|\mathcal{Y}^{\text{perm}}|$ 的选择未充分讨论，可能对 FG 性能有重要影响
4. **FG-ZS-SBIR 仅在 Sketchy 上评估**：缺少其他细粒度数据集（如 QMUL-Shoe/Chair）的结果
5. **推理速度未报告**：多模态 prompt 生成（尤其是 $\mathcal{B}_{vt}$ 需处理所有训练类）可能导致推理延迟
6. **损失权重 $\alpha, \beta$ 通过网格搜索确定**：对不同数据集的敏感性未分析

## 相关工作与启发

### 与 MaPLe (CVPR'23) 的对比
MaPLe 同为多模态 prompt learning，但仅支持视觉→文本的单向 token 共享（用文本 prompt 初始化视觉 prompt），且局限于特定层。SpLIP 实现了双向、全层的知识交换，在 SBIR 场景下更为有效。

### 与 CLIP-AT (CVPR'23) 的对比
CLIP-AT 使用视觉 prompt + patch shuffling 三元组目标。SpLIP 在三方面全面升级：多模态 prompt（vs 单模态）、条件拼图（vs 朴素排列匹配）、自适应 margin（vs 固定 margin）。跨数据集实验中 SpLIP 超过 CLIP-AT **14-15% mAP**。

### 启发
- **双向 prompt 共享的范式**可推广到其他跨模态任务（如 text-to-image 生成、VQA）
- **自适应 margin 思想**可用于任何基于 CLIP 语义空间的度量学习
- **条件拼图任务**思路可扩展到其他需要细粒度对齐的场景（如跨域 ReID、医学图像配准）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双向多模态 prompt 共享 + 条件拼图 + 自适应 margin，三个创新点协同工作
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三种 SBIR 设定 × 四个数据集 + 跨数据集泛化 + 详细消融，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，公式符号体系完整，图示直观
- **价值**: ⭐⭐⭐⭐ — 在草图检索领域树立了新的 CLIP 适配范式，双向 prompt 机制有较好的通用性
