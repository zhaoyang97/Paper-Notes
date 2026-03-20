# 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding

**会议**: ICCV 2025  
**arXiv**: [2412.18450](https://arxiv.org/abs/2412.18450)  
**代码**: [https://github.com/CognitiveAISystems/3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM)  
**领域**: 多模态VLM / 3D场景理解 / 场景图  
**关键词**: 3D Scene Graph, LLM, Semantic Relations, Visual Grounding, Learnable Graph Representation  

## 一句话总结
提出3DGraphLLM，首个将**3D语义场景图的可学习表示**直接输入LLM的方法——通过k近邻子图+三元组(object1, relation, object2)编码物体间语义关系，然后投影到LLM的token嵌入空间。在ScanRefer上Acc@0.5提升+6.4%（vs无语义关系的Chat-Scene），在Multi3DRefer上F1@0.5提升+7.5%，推理速度比GPT4Scene-HDM快5倍。

## 背景与动机
现有3D LMM用可学习表示编码场景物体时，通常只利用几何信息（3D坐标、点云特征），忽略了物体间的**语义关系**（如"桌子上的杯子"、"旁边的椅子"）。这些关系对指称接地（"桌子旁边的那个椅子"）和空间推理至关重要。虽然文本形式的场景图已被用于LLM（如ConceptGraphs），但文本描述一个物体需要几百个token，大画面下严重拖慢推理。3D场景图天然编码了物体和关系，但如何将其高效、可学习地输入LLM尚未探索。

## 核心问题
如何创建一种**高效且可学习的**3D语义场景图表示，使LLM能直接利用物体间的语义关系来提升3D视觉-语言任务的性能？

## 方法详解

### 整体框架
场景点云 → 实例分割(Mask3D/OneFormer3D) → 每个物体提取2D(DINOv2)+3D(Uni3D)特征 → VL-SAT生成物体间语义关系特征 → 三元组(obj_i, relation, obj_j)表示k近邻子图 → 投影层映射到LLM token空间 → LLM(LLAMA3-8B/Vicuna-7B + LoRA)回答用户查询

### 关键设计
1. **可学习场景图表示**：每个物体用其标识符token `<OBJ_i>`+ 2D特征$F_i^{2d}$+ k近邻子图描述。子图由三元组组成：$(F_i^v, F_{ij}^e, F_j^v)$——源物体3D特征、语义关系特征、目标物体3D特征。这比纯文本场景图紧凑得多（800 tokens vs 10400 tokens描述100个物体场景）

2. **语义边特征编码**：使用VL-SAT（基于CLIP知识迁移的3D场景图生成方法）从点云对中提取关系特征$Z_{ij}^e \in \mathbb{R}^{512}$，这是分类前的潜在特征，能捕获多种非互斥语义关系的组合

3. **k近邻+NMS+最小距离过滤**：完整场景图有$n(n-1)$条边，太多。只保留每个物体的k=2个最近邻。NMS过滤(IoU=0.99)去除重复物体，最小距离过滤(1cm)排除自身副本

4. **两阶段训练**：Stage 1用GT实例分割预训练投影层+LLM（高质量边特征）；Stage 2用Mask3D分割微调（适应噪声分割）

### 损失函数 / 训练策略
$$L(\theta) = -\sum_{i=1}^{\ell} \log P(s_i^{res} | s_{[1,...,i-1]}^{res}, s^{prefix})$$
- 4×A100, batch 8, 3 epochs, lr=5e-6, LoRA rank=16, cosine annealing
- 训练数据: ScanRefer+Multi3DRefer+Scan2Cap+ScanQA+SQA3D+RioRefer+3RQA (~370K)

## 实验关键数据

### 主要结果（Mask3D分割, LLAMA3-8B）
| 任务 | 数据集 | 指标 | 3DGraphLLM | Chat-Scene | 提升 |
|------|--------|------|------------|------------|------|
| 接地 | ScanRefer | Acc@0.5 | **56.6** | 50.2 | +6.4 |
| 接地 | Multi3DRefer | F1@0.5 | **59.9** | 52.4 | +7.5 |
| 描述 | Scan2Cap | C@0.5 | **81.0** | 77.1 | +3.9 |
| QA | ScanQA | CiDEr | 88.8 | 87.7 | +1.1 |
| QA | SQA3D | EM | 55.9 | 54.6 | +1.3 |

### 推理速度（Mask3D）
| 方法 | 每场景token数 | ScanRefer推理(s) |
|------|-------------|-----------------|
| GPT4Scene | 10400 | 1.9 |
| **3DGraphLLM** | **800** | **0.4** |

快4.75倍！

### 消融：语义关系的作用（GT分割, LLAMA3-8B）
| 边数 | ScanRefer Acc@0.5 | Multi3DRefer F1@0.5 |
|------|-------------------|---------------------|
| 0（无关系=Chat-Scene） | 61.5 | 64.4 |
| 2（+三元组） | **66.9** | **69.9** |

语义关系带来+5.4 / +5.5的显著提升。

### 消融要点
- **k=2最优**：在视觉接地/描述/QA三任务间取得最佳平衡
- **三元组 vs 扁平序列**：三元组表示(obj,rel,obj)优于仅序列化边特征（接地+1.0 F1@0.5）
- **两阶段训练**：GT预训练→Mask3D微调比直接在Mask3D上训练更好
- **ScanNet+3RScan预训练**：跨域数据进一步提升接地和QA
- **NMS+距离过滤**：解决Mask3D分割中的物体重复问题

## 亮点
- **首个可学习3D场景图→LLM**：填补了3D场景图与LLM之间的空白——不是用文本描述图，而是可学习的嵌入
- **极高效**：800 tokens描述100物体场景 vs Chat-Scene的200+GPT4Scene的10400，推理快5倍
- **语义关系的定量验证**：清晰证明加入物体间语义关系对接地任务有显著帮助（+5-7%）
- **与SOTA持平但更快**：与GPT4Scene-HDM质量相当但推理快5倍

## 局限性 / 可改进方向
- k增大时GPU内存消耗快速增长（k=4已是上限）
- 语义关系编码器(VL-SAT)在跨域时质量下降，需更鲁棒的关系提取
- n-gram指标不适合评估LLM的丰富输出（CIDEr 0分但描述正确的情况）
- 左右方向判断是常见失败模式

## 与相关工作的对比
- **Chat-Scene**：直接基线（无语义关系的物体列表+LLM）。3DGraphLLM加入场景图后在接地任务上提升6-7%
- **Robin3D**：用1M指令数据训练，3DGraphLLM仅用370K数据就达到可比性能
- **GPT4Scene-HDM**：质量相当但推理慢5倍
- **ConceptGraphs/BBQ**：用文本场景图+LLM，3DGraphLLM用可学习嵌入更紧凑

## 启发与关联
- 图结构→可学习token序列的转换思路可推广到其他图+LLM任务（知识图谱推理、分子图理解）
- VL-SAT的跨域关系提取能力证明了CLIP知识迁移对3D语义关系的有效性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个可学习3D场景图表示用于LLM，概念clean但核心组件是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 5个基准+丰富消融（边数/分割质量/子图表示/训练策略/可扩展性）
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰系统，消融设计全面
- 价值: ⭐⭐⭐⭐ 证明了语义关系对3D理解的重要性，高效推理对实际部署有价值
