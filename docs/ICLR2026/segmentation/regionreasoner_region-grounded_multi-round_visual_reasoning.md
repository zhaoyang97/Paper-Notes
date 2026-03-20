# RegionReasoner: Region-Grounded Multi-Round Visual Reasoning

**会议**: ICLR 2026  
**arXiv**: [2602.03733](https://arxiv.org/abs/2602.03733)  
**代码**: [RegionReasoner](https://github.com/wenfangsun/RegionReasoner)  
**领域**: segmentation / visual reasoning  
**关键词**: multi-round reasoning, region grounding, reinforcement learning, GRPO, VLM, referring segmentation  

## 一句话总结
提出 RegionReasoner，一个基于强化学习的多轮视觉推理框架，通过引用标注奖励和全局-局部一致性奖励，使推理轨迹必须显式引用参考区域坐标并保持语义连贯，在新构建的 RegionDial-Bench 上显著提升多轮定位和分割精度。

## 背景与动机
1. 现有 VLM 推理主要是单步或纯文本空间推理，缺乏迭代视觉上下文精炼能力
2. VisionReasoner 提供了单轮结构化推理但不跨轮传播区域引用
3. SegLLM 支持多轮交互分割但没有可验证的推理轨迹或 RL 信号
4. 朴素堆叠单轮推理导致：引用传播脆弱、坐标幻觉难以检测
5. 随着对话轮数增加，全局描述与局部证据语义漂移
6. 缺乏针对多轮推理精度和一致性的评估基准

## 方法详解
**结构化输出**: 每轮生成 4 个标签块 `<scene>` → `<focus>` → `<think>` → `<answer>`

**Reference-Grounded Thinking (引用标注推理)**:
- 推理轨迹 `<think>` 必须显式引用参考 bbox 坐标
- 引用奖励 $R_{ref}$：正确引用得分 + 幻觉坐标惩罚($\eta=0.5$)

**Global-Local Consistency Reward (全局-局部一致性)**:
- 从 `<scene>` 和 `<focus>` 提取关键词集合，与 `<think>` 计算非对称重叠
- 加入空间/比较/定位词汇先验 $\ell(h_t)$
- $R_{cons} = w_s \cdot \text{Ov}(s_t, h_t) + w_f \cdot \text{Ov}(f_t, h_t) + w_\ell \cdot \ell(h_t)$

**训练**: 基于 GRPO，Qwen2.5-VL-7B 初始化，4×H100 训练 ~10h

**RegionDial-Bench 基准**:
- 从 RefCOCO+/RefCOCOg 构建多轮对话
- RefCOCO+ Multi-turn: 715 图/2355 轮; RefCOCOg: 1580 图/4405 轮
- 支持检测(AP50)和分割(gIoU)的逐轮评估

## 实验关键数据
**7 轮检测 (RefCOCO+ Multi-turn AP)**:
| 方法 | R1 | R7 | Avg |
|------|-----|-----|-----|
| Qwen2.5-VL-7B | 65.5 | 25.9 | 49.9 |
| VisionReasoner-7B | 88.3 | 47.0 | 74.8 |
| **RegionReasoner-7B** | 89.3 | **64.7** | **80.7** |

- 后续轮次优势更大：R7 提升 +17.7 vs VisionReasoner
- RefCOCOg 上 Avg 75.0 vs VisionReasoner 73.6
- 分割任务同样优势显著
- 消融显示引用奖励主要减少坐标幻觉，一致性奖励稳定弱空间线索场景

## 亮点
- **可验证推理**: 推理轨迹中的 bbox 引用可自动检查和审计
- **两个互补奖励**: 引用奖励防幻觉 + 一致性奖励防语义漂移
- **多轮稳定性**: 后续轮次性能衰减显著小于基线
- **无额外任务头**: 检测和分割统一用 JSON `<answer>` 输出

## 局限性
- 基准规模较小（RefCOCO+ 仅 715 图），泛化性待验证
- 关键词匹配方式(lemma + 停用词移除)较粗糙，可能遗漏语义对齐
- 仅在 7B 规模验证，更大模型可能不需要如此结构化的约束
- 依赖约束解码保证格式，增加推理复杂度

## 相关工作
- **VisionReasoner**: 单轮结构化推理基线，本文扩展到多轮
- **SegLLM**: 多轮分割交互但无显式推理轨迹
- **Vision-R1/VLM-R1**: RL 增强 VLM 推理，本文加入区域标注约束
- **GRPO**: 训练策略优化算法

## 评分
- 新颖性: ⭐⭐⭐⭐ (引用标注推理 + 一致性奖励)
- 实验充分度: ⭐⭐⭐⭐ (检测+分割 + 逐轮分析 + 消融)
- 写作质量: ⭐⭐⭐⭐ (形式化完整)
- 价值: ⭐⭐⭐⭐ (多轮视觉推理新方向)
