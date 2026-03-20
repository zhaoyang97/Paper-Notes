# HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2506.07296](https://arxiv.org/abs/2506.07296)  
**代码**: 无  
**机构**: Leiden University, Booking.com  
**领域**: 多模态检索 / 信息检索 / 旅行搜索  
**关键词**: 多模态稠密检索, 非对称编码器, 多任务优化, 酒店搜索, SLM-LLM联合训练  

## 一句话总结
提出 HotelMatch-LLM，用 SLM 编码 query + LLM 编码酒店文档的非对称架构，配合三目标多任务优化（检索对齐 + MLM地理预测 + 视觉设施识别）和 patch 级 mean pooling 多图处理，在旅行领域多模态检索任务上显著超过 MARVEL/VISTA 等 SOTA。

## 背景与动机
当前在线旅行搜索平台（如 Booking.com）依赖预定义筛选器：用户必须先选国家/城市，再设置价格、星级等参数。这种方式无法处理自然语言描述的复杂需求，比如"有泳池和海景的精品酒店"或"靠近地铁站的三星级以上酒店"。已有的多模态检索模型（MARVEL、VISTA）虽然在 web 搜索上表现不错，但只支持单张图片输入，无法处理酒店场景中动辄几十上百张图片的图库。同时，LLM 做 query embedding 的在线推理成本太高，不适合日均数百万查询的生产环境。

## 核心问题
1. **如何在旅行领域支持自然语言多模态检索？** 传统的关键词+筛选器模式不支持 free-form query
2. **如何高效处理大量属性图片？** 酒店最多有 306 张图片，现有方法只能处理单张
3. **如何平衡检索效果与在线推理效率？** LLM 做 embedding 效果好但太慢

## 方法详解

### 整体框架
HotelMatch-LLM 是一个非对称双编码器检索架构。Query 侧用小模型（GTR-Base-110M）编码，文档侧用大模型（GTR-Large-335M）编码酒店的文本+视觉信息，通过 cosine similarity 计算相关性。训练时同时优化三个目标：检索对齐、MLM 地理信息预测、视觉设施识别。

**输入**：query 文本 → SLM 编码得到 query embedding；酒店图片（多张）→ CLIP 编码 + mean pooling + 线性映射得到视觉 tokens，拼接酒店文本描述 → LLM 编码得到文档 embedding。

### 关键设计

1. **多图 Patch Mean Pooling**：
   每张图片通过 CLIP 视觉编码器提取 49 个 patch embedding（224×224 图片 + 32×32 window），然后对所有图片的对应 patch 位置做 mean pooling，得到固定尺寸的 (49 × dim) 表示。这样不管酒店有多少张图片，最终都压缩成 49 个视觉 token。再通过线性层投影到 LLM 的 token embedding 空间，与文本 token 拼接后输入 LLM。这个设计理论上可以处理**无限数量**的图片。

2. **SLM-LLM 非对称架构**：
   核心洞察是酒店数据比 query 复杂得多，需要更大的模型来表示。因此用 SLM（GTR-Base, 110M）做在线 query embedding，用 LLM（GTR-Large, 335M）做离线文档 embedding。SLM 输出通过线性层投影到与 LLM 相同的维度空间。关键细节是 SLM 和 LLM 需要使用**不同的学习率**（SLM: 5e-4, LLM: 5e-6），共享学习率会导致效果急剧下降。

3. **领域特定多任务优化**：
   - **检索损失 L_Ret**（权重 0.7）：标准对比学习，softmax + cross-entropy，拉近 query 与正样本文档的余弦相似度
   - **MLM 损失 L_MLM**（权重 0.2）：遮掩酒店描述中的城市和国家 token，预测地理信息，增强模型对地理特征的理解
   - **视觉设施识别损失 L_VisF**（权重 0.1）：基于文档 embedding 预测 120 种设施（泳池、健身房、阳台等）是否存在，标签来自 MUMIC 方法自动识别。Binary cross-entropy 损失
   - 权重通过 grid search（以 0.1 为步长）在验证集上确定

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_{final} = 0.7 \cdot \mathcal{L}_{Ret} + 0.2 \cdot \mathcal{L}_{MLM} + 0.1 \cdot \mathcal{L}_{VisF}$
- 训练标签由 GPT-4o 生成（query-hotel 对的二元相关性标注），与人工标注有强相关性
- 训练 10 个 epoch，early stopping（5 步验证集无提升）
- 评估采用先 CLIP 初检索 top-100，再 re-ranking 的流程

## 实验关键数据

| 测试集 (指标) | Real-world MRR/nDCG | Vision-driven MRR/nDCG | Text-driven MRR/nDCG | OOD MRR/nDCG |
|---|---|---|---|---|
| BM25 | .506/.401 | .138/.195 | .798/.825 | .588/.489 |
| MARVEL (Fine-tuned) | .603/.503 | .219/.326 | .810/.833 | .660/.515 |
| VISTA (Fine-tuned) | .582/.465 | .216/.321 | .802/.839 | .662/.513 |
| **HotelMatch-LLM** | **.681/.600** | **.247/.362** | **.863/.884** | **.704/.558** |

- 主测试集 MRR@10：0.681 vs MARVEL 0.603 → **提升 12.9%**
- 在所有四类测试集上均统计显著优于所有 baseline（paired t-test, p<0.05, Bonferroni 校正）
- Full-ranking（3.1M 文档）MRR：0.675 vs MARVEL 0.589

**效率对比**：
| 模型 | 推理延迟 (ms) | MRR | nDCG |
|---|---|---|---|
| VISTA | 16.17 | .572 | .465 |
| MARVEL | 31.07 | .603 | .503 |
| HotelMatch-LLM | 18.69 | .681 | .600 |

推理延迟仅比 VISTA 多 2.5ms，但效果大幅领先；比 MARVEL 快 1.7 倍。

### 消融实验要点
- **多任务消融**：去掉 MLM 掉点最多（MRR .681→.650，-4.6%），说明地理信息理解对酒店检索至关重要。去掉 VisF 也有明显下降（→.664），两个都去掉 → .632
- **多图方法对比**：mean pooling over patches（无限图片）> 1TPI-Patch（.672, 限 50 张）> 1TPI-CLS（.652, 限 50 张）
- **LLM 骨干泛化性**：用 Zeta-Alpha-E5-Mistral-7B 做文档编码器时 MRR 达 .719，Stella-en-1.5B 达 .694，更大的 LLM 带来更好的酒店表示
- **学习率**：SLM 和 LLM 分离学习率（5e-4 / 5e-6）MRR=.681，共享学习率只有 .315，差距极大
- **去掉视觉**：HotelMatch-LLM w/o vision MRR .595，说明视觉信息贡献很大

## 亮点
- **非对称架构思路很实用**：SLM 做在线 query、LLM 做离线文档，在工业级系统中非常合理且易部署。延迟仅 18.69ms，接近最轻量的 VISTA
- **Patch-level mean pooling 处理多图的方案简洁优雅**：不受图片数量限制，避免了 token 数量爆炸问题。实验中最多处理 306 张图片
- **三目标多任务设计有领域针对性**：MLM 做地理预测、VisF 做设施识别，都是酒店检索场景下的 domain-specific 信号，不是随便加个辅助任务
- **分离学习率的细节值得注意**：SLM 和 LLM 联合训练时用不同 LR 是关键，共享 LR 直接导致性能崩塌

## 局限性 / 可改进方向
- **依赖 GPT-4o 生成训练标签**：如果合成标注有偏差会影响模型质量，且无法复现
- **不支持多模态 query**：当前只支持纯文本 query，不能处理"找类似这张图片的酒店"这种带图 query
- **缺乏个性化**：没有考虑用户历史偏好和交互行为
- **数据集不公开**：HotelMatch 数据集来自 Booking.com 内部，无法复现验证
- **SLM/LLM 骨干相对较小**：主实验用的 GTR 模型最大只有 335M，和当前主流的 7B+ 模型差距明显，虽然泛化实验用了 7B 但没有充分探索

## 与相关工作的对比
- **vs MARVEL**（ACL 2024）：MARVEL 是多模态检索 SOTA，但只能处理单张图片、使用对称编码器架构。HotelMatch-LLM 在所有测试集上显著超过 MARVEL，且推理快 1.7 倍
- **vs VISTA**（ACL 2024）：VISTA 也只支持单图，且在本文的领域场景下效果不如 MARVEL。HotelMatch-LLM 效果全面领先，延迟仅多 2.5ms
- **vs CLIP**：Zero-shot CLIP 在领域数据上效果一般，多模态版本也没有大幅提升。说明领域微调（domain-specific fine-tuning）对旅行搜索至关重要

## 启发与关联
- 非对称编码器设计与 [非对称跨模态知识蒸馏 idea](../../ideas/model_compression/20260316_asymmetric_multimodal_scaling.md) 高度相关：HotelMatch-LLM 用经验验证了"query 简单 → 小编码器足够、文档复杂 → 大编码器必要"这一假设，可作为该 idea 的实证支持
- Patch-level mean pooling 的多图聚合思路可迁移到其他多图场景（商品检索、房产搜索、医学多视角影像等）
- 分离学习率联合训练 SLM+LLM 的 trick 对一切非对称架构都有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 非对称架构+领域多任务+多图处理三个组件各自不算全新，但组合有意义
- 实验充分度: ⭐⭐⭐⭐ 四类测试集+多方位消融+泛化性+效率分析，但数据集不公开是硬伤
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式规范，实验逻辑完整
- 价值: ⭐⭐⭐⭐ 工业实用性强，非对称架构和多图处理的经验对 IR 社区有参考价值
