# mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2409.03420](https://arxiv.org/abs/2409.03420)  
**代码**: [https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl2](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl2)  
**领域**: 多模态VLM / 文档理解  
**关键词**: 文档理解, 视觉token压缩, 多页文档, OCR-free, 高分辨率  

## 一句话总结
提出High-resolution DocCompressor模块，利用低分辨率全局视觉特征作为query通过交叉注意力将高分辨率文档图像压缩为仅324个token（不到同类方法的20%），在多页文档理解benchmark上达到SOTA且首token延迟降低50%+。

## 背景与动机
OCR-free文档理解的MLLM通过增加分辨率获得了强大性能（如InternVL2在DocVQA上91.6），但代价是单张文档图像生成数千个视觉token（InternVL2约3133个）。这导致：(1) GPU内存占用过大；(2) 推理延迟高；(3) 多页文档几乎不可行。已有压缩方法要么独立压缩每个裁剪块（总token仍多），要么用可学习query（忽略布局信息）。

## 核心问题
如何将文档图像大幅压缩到极少token（<400个），同时保留布局结构和文本信息？

## 方法详解

### 整体框架
形状自适应裁剪 → 低分辨率ViT编码 → H-Reducer对齐到LLM维度 → **High-resolution DocCompressor**交叉注意力压缩 → LLM多模态理解。三阶段训练：单图预训练→多图续训→多任务微调。

### 关键设计

1. **High-resolution DocCompressor**: 核心创新。全局低分辨率图像经ViT+H-Reducer得到的特征图作为cross-attention的**query**，高分辨率子图拼接的特征图作为**key/value**。每个query自然对应原图的一个布局区域，只attend相同相对位置的高分辨率特征——而非全图attend，既保留布局感知又降低计算复杂度。最终将任意分辨率文档压缩为固定的18×18=324个token。

2. **布局感知压缩**: 关键insight——文档中同一布局区域的文本语义连贯（如双栏论文中"相关工作"一栏的文本），按布局区域做局部压缩比全局压缩更有效。全局特征隐式编码了布局信息，作为query天然地实现了布局感知。

3. **文本对齐后再压缩**: 另一个insight——在V2T模块（H-Reducer）之后做压缩，而非直接压缩ViT输出。因为V2T已把视觉特征对齐到LLM的文本空间，此时压缩类似于"文本摘要"（与Gist Token思路类似），比纯视觉压缩更能保留文本语义。

4. **三阶段训练**: Stage 1（单图预训练，DocStruct4M 4M数据）训练编码器+DocCompressor；Stage 2（多图续训，1.6M数据）训练多页理解能力；Stage 3（多任务微调）在下游任务上精调。

### 损失函数 / 训练策略
- 标准自回归文本生成损失
- Stage 1: lr=1e-4, batch=1024, 12k steps
- Stage 2: lr=2e-5, batch=1024, 2.4k steps  
- Stage 3: lr=2e-5, batch=256, 9k steps

## 实验关键数据

**单页文档理解（324 tokens vs 1k+ tokens基线）**:

| 模型 | Token数 | DocVQA | InfoVQA | ChartQA | TextVQA |
|------|---------|--------|---------|---------|---------|
| InternVL2 (8B) | ~3133 | **91.6** | **74.8** | **83.3** | 77.4 |
| DocOwl 1.5 (8B) | ~1698 | 82.2 | 50.7 | 70.2 | 68.6 |
| **DocOwl2 (8B)** | **324** | 80.7 | 46.4 | 70.0 | 66.7 |
| TextMonkey (9B) | 768 | 73.0 | 28.6 | 66.9 | 65.9 |
| QwenVL (9B) | 256 | 65.1 | 35.4 | 65.7 | 63.8 |

→ DocOwl2用324 token达到了DocOwl 1.5用1698 token的98%性能

**多页文档理解**:
- MP-DocVQA: DocOwl2 SOTA
- DUDE: DocOwl2 SOTA
- 首Token延迟(FTL)降低>50%

### 消融实验要点
- **压缩位置**: V2T之后压缩 > ViT之后压缩（DocVQA差4.7%），验证了"文本对齐后再压缩"的假设
- **Query来源**: 全局特征做query > 可学习query > 选择性token做query
- **压缩层数**: 2层交叉注意力最优
- **局部attend vs 全局attend**: 局部attend（布局感知）优于全局attend
- **三阶段训练**: 每个阶段都有贡献，特别是多图续训对多页理解至关重要

## 亮点
- **极致的token效率**: 324 token vs 3000+ token，减少90%以上，且性能仅降很少
- **布局感知是核心**: 利用全局低分辨率特征做query，自然实现按布局区域压缩
- **V2T后压缩的insight**: 将视觉压缩转化为"文本摘要"，效果显著优于直接压缩视觉特征
- **多页文档成为可能**: 324 token/页使得20+页的完整文档理解在单张GPU上可行

## 局限性 / 可改进方向
- 单页性能仍低于不压缩的高token方法（InternVL2: 91.6 vs DocOwl2: 80.7 on DocVQA）
- 324是固定的压缩目标，未探索自适应压缩比（简单页面可更低，复杂页面应更高）
- DocCompressor增加了额外的模型参数和训练成本
- 对极小文字或复杂表格的压缩可能丢失关键信息
- 仅在文档场景验证，对自然图像的适用性未测试

## 与相关工作的对比
- **vs InternVL2**: InternVL2更准但用3133 token，DocOwl2用324 token效率高10倍
- **vs TokenPacker**: TokenPacker独立压缩每个crop，总token仍多（467-1833）；DocOwl2全局布局感知压缩更高效
- **vs TextMonkey**: TextMonkey选择性筛选token（768个），信息可能覆盖不全；DocOwl2全区域覆盖

## 启发与关联
- "V2T之后再压缩"的insight与Gist Token研究互补——两者都发现在文本空间做压缩更有效
- 自适应压缩比是一个重要改进方向——简单页面的布局信息已经足够压缩到更少token
- DocCompressor的布局感知交叉注意力可以迁移到VLM的视频帧压缩（帧=页面）

## 评分
- 新颖性: ⭐⭐⭐⭐ 布局感知压缩和V2T后压缩的两个insight都有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 10个单页+2个多页benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐ 图示清晰，不同压缩方法的对比很直观
- 价值: ⭐⭐⭐⭐⭐ 使多页文档OCR-free理解成为实际可行，工业价值高
