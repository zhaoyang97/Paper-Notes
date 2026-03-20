# Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2312.06109](https://arxiv.org/abs/2312.06109)  
**代码**: [https://github.com/Ucas-HaoranWei/Vary](https://github.com/Ucas-HaoranWei/Vary)  
**领域**: 多模态/VLM  
**关键词**: vision vocabulary, LVLM, document OCR, chart understanding, fine-grained perception  

## 一句话总结
提出 Vary 方法，通过生成并融合新的视觉词汇表（vision vocabulary）来扩展 LVLM 的视觉感知能力，使模型在保持原有通用能力的同时，获得文档级 OCR、图表理解等细粒度视觉感知新能力。

## 背景与动机
当前主流 LVLM（如 BLIP-2、MiniGPT-4、LLaVA、Qwen-VL 等）几乎都依赖同一个视觉词汇表——CLIP-ViT。CLIP 通过 4 亿+ 图文对的对比学习训练，能覆盖大部分自然图像和常见视觉任务。然而，对于一些特殊的视觉任务（如文档级 OCR、图表理解，尤其是非英文场景），CLIP 的视觉词汇表存在以下问题：

1. **编码效率低**：CLIP 难以将文档/图表中的密集文本信息高效编码到固定数量（通常 256 个）的 token 中
2. **视觉 OOV 问题**：这些特殊场景的图像对 CLIP 来说如同"外语"，产生视觉词汇表外（out-of-vocabulary）问题
3. **已有解法的局限**：mPlug-Owl 和 Qwen-VL 通过解冻 CLIP 来缓解，但可能覆写原有知识、训练效率低、且因 LLM 强记忆能力不能多 epoch 训练

核心类比：在 NLP 领域，将英文 LLM 迁移到中文时需要扩展文本词汇表来提升编码效率；同理，对于 CLIP 不擅长的视觉"外语"图像，也需要扩展视觉词汇表。

## 核心问题
如何简单高效地为 LVLM 扩展视觉词汇表，使其在不破坏原有通用能力的前提下，获得细粒度文档/图表感知等新能力？

## 方法详解

### 整体框架
Vary 的流程分为两个阶段：

1. **视觉词汇表生成（Vary-tiny）**：训练一个小型管道（vocabulary network + OPT-125M）通过自回归方式生成新的视觉词汇表
2. **视觉词汇表融合（Vary-base）**：将新词汇表与原 CLIP 词汇表并行融合，赋予 LVLM 新能力

### 关键设计

1. **新视觉词汇表网络**：采用 SAM 预训练的 ViTDet（base 规模）作为主干。输入分辨率 1024×1024，输出步长 16，最后层特征为 64×64×256。在其后添加两个卷积层进行 token 合并：第一个 conv（kernel=3）将特征转为 32×32×512，第二个 conv 进一步转为 16×16×1024，展平后得到 256×1024，与 CLIP-ViT 的输出形状对齐。

2. **自回归式词汇表训练（Vary-tiny）**：用 vocabulary network + OPT-125M 小模型组成 Vary-tiny。训练数据包括：
   - **正样本**：文档数据（1M 中文 + 1M 英文页面，从 arXiv 和 CC-MAIN PDF 提取）；图表数据（matplotlib 渲染 250k×2 + pyecharts 渲染 500k×2，中英文各半，标注为 python-dict 格式）
   - **负样本**：COCO 120k 自然图像，文本标注为固定简单句（如"It's an image of nature"），确保新词汇表在自然图像上不产生噪声

3. **双词汇表并行融合（Vary-base）**：新旧两个视觉词汇表各配独立的 linear 输入嵌入层（输入 1024 → 输出 2048），两路 token 拼接后通道为 4096，与 LLM（Qwen-7B 或 Vicuna-7B）的输入维度对齐。融合阶段**冻结**两个视觉词汇表网络，仅训练输入嵌入层和 LLM。

4. **高质量合成数据**：为 Vary-base 额外制作了：
   - LaTeX 渲染文档（0.5M 英文 + 0.4M 中文，支持公式/表格，标注为 Mathpix markdown 格式）
   - 语义关联图表（用 GPT-4 生成有语义关联的图表内容，额外渲染 200k）
   - 通用数据：4M LAION-COCO 图文对（预训练）+ LLaVA-80k/665k + DocVQA/ChartQA 训练集（SFT）

### 损失函数 / 训练策略
- **Vary-tiny 训练**：标准自回归 next-token prediction loss。batch size 512，训练 3 epochs，AdamW 优化器 + cosine annealing，lr=5e-5
- **Vary-base 训练**：两阶段——预训练（lr=5e-5，batch 256，1 epoch）+ SFT（lr=1e-5，batch 256，1 epoch）。冻结所有视觉词汇表网络权重，仅训练输入嵌入层和 LLM
- 输入格式：`<img>"<image>"</img> "text"`，自回归输出文本

## 实验关键数据

| 数据集/任务 | 指标 | Vary-base | 对比方法 | 备注 |
|------------|------|-----------|----------|------|
| DocVQA (test) | ANLS | **78.2%** | Qwen-VL 65.1%, Pix2Struct 72.1% | 大幅超越 |
| DocVQA (val) | ANLS | **76.3%** | - | - |
| ChartQA (avg) | Relaxed Acc | **66.1%** (665k) | Qwen-VL 65.7%, Matcha 64.2% | 可比/微超 |
| ChartQA (human) | Relaxed Acc | **43.8%** | Matcha 38.2% | +5.6% |
| ChartQA (aug) | Relaxed Acc | **88.3%** | Matcha 90.2% | 略低 |
| MMVet (total) | Score | **36.2%** (Qwen-7B) | LLaVA-13B 32.9%, LLaVA1.5-7B 30.5% | 超越更大模型 |
| 英文文档 OCR | Edit Distance | **0.106** | Nougat 0.126 | 更优 |
| 英文文档→Markdown | Edit Distance | **0.181** | Nougat 0.245 | 大幅改善 |
| 英文文档→Markdown | F1 | **81.10%** | Nougat 79.97% | +1.13% |
| 中文文档 OCR | Edit Distance | 0.174 | - | 独有能力 |

### 消融实验要点
- **Vary-tiny vs Vary-base**：Vary-tiny（仅 OPT-125M）已具备中英文 OCR 能力（中文 Edit Distance 0.266，英文 0.197），验证了新视觉词汇表的有效性；升级为 7B LLM 后性能进一步提升
- **负样本的作用**：使用 COCO 自然图像作为负样本训练，确保新词汇表在自然图像上不产生噪声，从而不伤害模型通用能力
- **Vicuna vs Qwen LLM**：相同设置下 Vary-base (Vicuna-7B, 665k) 在 MMVet 达 32.9%，与 LLaVA-13B 持平；换用 Qwen-7B 后达 36.2%，说明 LLM 的中文能力对整体效果重要
- **SFT 数据量**：LLaVA-80k vs LLaVA-665k 在 ChartQA 上从 65.3% 提升到 66.1%，DocVQA 保持稳定（78.2% vs 78.1%）

## 亮点
- **视觉词汇表扩展的新范式**：首次从"扩展视觉词汇表"的角度思考 LVLM 能力增强，类比 NLP 中文本词汇表扩展，思路清晰且有启发性
- **自回归式词汇表生成**：相比 CLIP 的对比学习，自回归方式更适合密集感知任务——能压缩更长文本、支持更多样的数据格式（如带 prompt 的 VQA 数据）
- **解耦式训练策略**：先独立训练新词汇表（小模型低成本），再冻结融入大模型，避免知识覆写，训练效率高
- **强大的合成数据引擎**：构建了大规模中英文文档和图表合成数据管线（PDF 提取 + LaTeX 渲染 + pyecharts/matplotlib 渲染），可复用价值高
- **中文能力**：具备中文文档 OCR 和中文图表理解能力，在当时是稀缺特性

## 局限性 / 可改进方向
1. **分辨率限制**：虽然新词汇表网络接受 1024×1024 输入，但对更高分辨率（如 2K/4K 文档）的处理仍有限
2. **词汇表数量固定**：当前方案仅扩展一个新视觉词汇表，对于多种不同类型的"视觉外语"任务，可能需要更灵活的扩展机制
3. **评估范围有限**：主要评测了文档和图表任务，对其他可能受益于词汇表扩展的场景（如遥感、医学影像）未做验证
4. **固定 token 数量**：新旧词汇表各输出 256 个 token，总共 512 个 token，对于超长文档可能仍不够
5. **自回归小模型的瓶颈**：Vary-tiny 使用 OPT-125M 作为 decoder，其容量可能限制了视觉词汇表的质量上限

## 与相关工作的对比
| 方法 | 视觉词汇表 | 是否冻结 | 特殊能力 | LLM |
|------|-----------|---------|---------|-----|
| LLaVA | CLIP-L | 冻结 | 通用VQA/对话 | Vicuna |
| BLIP-2 | CLIP + Q-Former | 冻结 | 通用VQA/对话 | OPT/FlanT5 |
| Qwen-VL | CLIP-G + 交叉注意力 | 解冻 | +OCR/定位 | Qwen-7B |
| mPlug-Owl | CLIP-L | 解冻 | +通用增强 | LLaMA |
| Nougat | Swin Transformer | - | 文档解析专用 | mBART decoder |
| **Vary** | **CLIP-L + 新词汇表** | **双冻结** | **+文档OCR+图表理解** | **Qwen-7B/Vicuna** |

Vary 的独特之处在于不修改原有 CLIP，而是额外增加一条视觉通路，用"加法"而非"改写"的方式扩展能力。

## 启发与关联
1. **视觉词汇表的类比思维**：将 NLP 的词汇表扩展思路迁移到视觉侧，暗示 LVLM 的能力瓶颈可能不仅在 LLM 端，视觉编码器的表达能力同样关键
2. **合成数据的重要性**：Vary 验证了大规模合成数据（文档渲染、图表渲染）在特定视觉任务中的有效性，为数据匮乏场景提供了思路
3. **与 Vary 后续工作的关联**：Vary 系列后续发展出 Vary-toy（更小规模）和 Mini-Monkey 等变体，在文档理解方向持续推进
4. **对多视觉编码器融合的启示**：Vary 的双编码器并行 + 冻结融合策略，为后续 InternVL、Cambrian 等多编码器 LVLM 提供了参考

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次从视觉词汇表扩展角度优化 LVLM，思路清晰且有启发性
- 实验充分度: ⭐⭐⭐⭐ 涵盖文档 OCR、DocVQA、ChartQA、MMVet 等多个维度，消融较充分
- 写作质量: ⭐⭐⭐⭐ 行文流畅，类比贴切，结构清晰
- 价值: ⭐⭐⭐⭐⭐ 开创了视觉词汇表扩展的新研究方向，后续影响力大（Vary 系列、多编码器 LVLM 等）
