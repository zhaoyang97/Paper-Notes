# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding

**会议**: NeurIPS 2025
**arXiv**: [2505.18079](https://arxiv.org/abs/2505.18079)
**代码**: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)
**领域**: LLM Agent / 视频理解
**关键词**: video understanding, agentic search, tool use, long-form video, multi-granular database, adaptive workflow

## 一句话总结
提出 DVD（Deep Video Discovery）agent，将长视频理解建模为多步信息搜索问题：先将长视频构建为多粒度结构化数据库（全局摘要 + clip 级字幕嵌入 + 帧级像素），再提供三种搜索工具（Global Browse / Clip Search / Frame Inspect），由 reasoning LLM 通过 observe-reason-act 循环自主编排搜索轨迹，在 LVBench 达 74.2%（超先前 SOTA MR.Video 13.4 pp），加字幕 76.0%。

## 背景与动机
- 长视频（小时级）信息密度极高，即使百万 token 上下文的 LLM 也无法直接处理；且随上下文增长，指令跟随与推理能力下降
- 先前 video agent（VideoTree、VCA）采用**固定工作流**——如树搜索从根到叶、固定的"预测-反思-搜索-合并"循环——无法针对不同查询自适应选择策略
- 核心洞察：受 Deep Research/Deep Search 启发，将长视频理解看作**多步信息搜索问题**，视频是待探索的环境，片段是信息单元，agent 自主规划搜索路径

## 方法详解

### 阶段一：多粒度视频数据库构建（离线）

**时间分段**：将长视频 V 均匀切分为 N = ⌈len(V)/t⌉ 个不重叠 clip，t=5 秒，每 clip 以 2fps 解码为帧序列

**三级信息提取**：
1. **帧级（Frame Level）**：保留原始解码帧 f_i，按 clip 索引存储，供后续像素级细节分析
2. **Clip 级（Clip Level）**：用 VLM（GPT-4.1）为每个 clip 生成文本描述 c_i，再用语言嵌入模型编码为向量 e_i，用于语义检索
3. **全局级（Global Level）**：在逐 clip 生成描述的过程中，维护一个**渐进式主体注册表 S**——每当出现新主体（人物、物品等），记录其名称、外观、身份、动作和时间跨度。最终 S_N 构成全局主体索引

**最终数据库**：$\mathcal{D} = \{S, \{f_i, c_i, e_i\}_{i=1}^{N}\}$——文本可检索 + 像素可回溯的结构化表示

### 阶段二：Agentic Search and Answer（ASA）

**三种搜索工具**：

| 工具 | 粒度 | 输入 | 输出 | 用途 |
|------|------|------|------|------|
| Global Browse | 全局 | 数据库 D + 用户查询 Q | 主体摘要 + 事件摘要 | 获取全局上下文；主体摘要来自预构建的注册表，事件摘要通过均匀采样帧 + VLM 生成 |
| Clip Search | Clip 级 | 数据库 D + agent 合成查询 Q̂ + top-k | top-k 相关 clip 及其描述 | 通过 cosine 相似度在嵌入空间中检索相关片段；agent 可迭代调用、逐步细化查询 |
| Frame Inspect | 帧级 | 数据库 D + agent 合成查询 Q̂ + 时间范围 [t_s, t_e] | VQA 回答 | 加载原始帧（最多 50 帧）+ VLM 做开放式 VQA，获取细粒度视觉信息 |

**Agent 设计——observe-reason-act 循环（类 ReAct）**：
- 动作空间 A = {Global Browse, Clip Search, Frame Inspect, Answer}
- 每一步：LLM 根据历史 H_i 推理 → 选择动作 A_i 和参数 P_i → 获取观察 O_i → 更新历史
- 终止条件：选择 Answer 动作或达到最大步数 N=15
- **关键设计决策**：不手动指定工具使用模式或搜索策略，完全依靠 LLM 的推理能力自主编排

### 实现细节
- 数据库构建 VLM：LVBench 用 GPT-4.1，其他基准用 GPT-4.1-mini（降成本）
- 推理模型 M_reasoning：OpenAI o3（也用于 Frame Inspect 的 VQA）
- Clip Search 默认 top-k=16，LLM 可自行调整
- 帧分辨率统一 resize 到 720p
- 字幕加持版本：用 WhisperX 做 ASR，转录文本指导分段并丰富描述

## 实验关键数据

### LVBench（1549 题 / 103 个小时级视频）
| 方法 | 准确率 |
|------|--------|
| GPT-4o | 48.9% |
| OpenAI o3（256帧直接输入） | 57.1% |
| MR. Video（上一个 SOTA agent）| 60.8% |
| **DVD（ours）** | **74.2%** |
| DVD + 字幕 | **76.0%** |

- 超 MR.Video 13.4 pp，超 VCA 32.9 pp，超 base VLM o3 17.1 pp

### 其他基准
- LongVideoBench Long subset：68.6%（超先前 SOTA 7.0 pp）
- Video MME Long：67.3%（超 AdaRETAKE 2.3 pp）
- EgoSchema：76.6%（超人类水平 ~76%）

### 消融研究

**模型选择影响（Table 4）**：
- 推理模型最关键：o3 → o4-mini 掉 5.8 pp，o3 → GPT-4o 掉 13.7 pp
- 数据库构建 VLM：GPT-4.1 → 4.1-mini 仅掉 4.1 pp
- Frame Inspect VLM：o3 → 4.1-mini 掉 3.7 pp

**工具消融（Table 5）**：
- 去掉 Clip Search 最致命（-12.3 pp）——核心搜索能力
- 去掉 Frame Inspect 掉 8.4 pp——细粒度理解依赖
- 去掉 Global Browse 掉 2.9 pp——全局语境辅助

**开源模型兼容性（Table 6）**：
- DeepSeek-R1 做推理模型：68.5%（仍超所有先前方法）
- Qwen3-32B：57.3%（32B 模型即超 GPT-4o 和 o3 直接输入）

**自适应 vs 固定工作流（Table 7）**：
- 固定 VideoAgent 工作流平均 11.1 步仅 70.2%；DVD 自适应平均 7.3 步达 74.2%——步数更少、效果更好

## Agent 行为模式分析（核心贡献之一）

论文将 agent 的工具调用行为分为五类并分析：
1. **Global Browse Only**：一次全局浏览即回答，罕见但准确率极高
2. **Simple Action**：直接 search → inspect → answer，最常见（>50%），准确率高
3. **Iterative Search**：多轮交替 Clip Search 和 Frame Inspect，步数更长（~8步），准确率略低
4. **Frame Inspect Trap**：连续 3+ 次 Frame Inspect 陷入细节循环，准确率显著下降
5. **Clip Search Trap**：连续 3+ 次 Clip Search 反复搜索同类信息，o3 的主要失败模式

**两个关键洞察**：
- **推理长度的双重性**：同一模型内，越长的推理轨迹通常意味着越不确定、准确率越低；但跨模型比较时，能进行更深推理的模型反而表现更好
- **过度自信导致行为坍缩**：GPT-4o 91.4% 的查询都用 Simple Action 模式（平均仅 4.6 步），过早结论、很少探索替代策略，是其表现差的根本原因

## 亮点
- **自主搜索范式**：不预定义工作流，让 LLM 自己决定搜索轨迹——更接近人类分析视频的方式
- LVBench 74.2% 是决定性的领先（超上一个 SOTA 13.4 pp）
- 多粒度数据库设计精巧：文本可检索 + 像素可回溯，兼顾效率和精度
- Agent 行为模式分析提供了 video agent 设计的实用洞察
- 开源模型 DeepSeek-R1 也能达到 68.5%，证明框架的通用性

## 局限性
- 迭代推理引入较高计算开销（多轮 LLM + VLM 调用）
- 高度依赖推理模型能力——弱模型（如 GPT-4o）会出现行为坍缩
- 工具集固定为三种，可扩展性有限（如缺少 OCR/ASR 专用工具）
- Azure 内容过滤误判部分基准数据导致少量性能损失

## 与相关工作的对比
- **vs VideoTree / VCA**：固定树搜索策略 → DVD 自适应搜索，LVBench 上超 VCA 32.9 pp
- **vs MR.Video**：先前最佳 agent 60.8% → DVD 74.2%，核心差异在于 DVD 不指定工具顺序
- **vs AdaRETAKE**：视觉 token 压缩方案 53.3% → DVD 74.2%，agent 搜索显著优于压缩策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Deep Search 理念迁移到视频理解，agent 自主编排搜索工作流
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个基准 + 详细消融 + 行为模式分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，行为分析有洞察
- 价值: ⭐⭐⭐⭐⭐ 长视频理解新范式，数据决定性领先，开源可复现
