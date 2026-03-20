# Experience-based Knowledge Correction for Robust Planning in Minecraft

**会议**: ICLR 2026  
**arXiv**: [2505.24157](https://arxiv.org/abs/2505.24157)  
**代码**: 无  
**领域**: AI安全 / Agent 鲁棒规划  
**关键词**: LLM planning, knowledge correction, Minecraft, embodied agent, self-correction failure  

## 一句话总结
证明 LLM 无法通过 prompting 自我纠正其错误的规划先验知识（物品依赖关系），提出 XENON——通过算法化的知识管理（自适应依赖图 ADG + 失败感知动作记忆 FAM）从二值反馈中学习，使 7B LLM 在 Minecraft 长期规划中超越使用 GPT-4V + oracle 知识的 SOTA。

## 研究背景与动机
1. **领域现状**：LLM 驱动的 Agent 在 Minecraft 等长期规划任务中需要准确的物品依赖知识（如钻石镐需要钻石+木棍），但 LLM 的参数化知识常有错误。
2. **现有痛点**：自我纠正（self-correction）——即用 prompt 让 LLM 反思并修正知识——在参数化知识错误上无效。LLM 会反复犯同样的错误，因为错误编码在权重中，prompt 无法改变。
3. **核心矛盾**：LLM 的语言理解能力强但事实知识不可靠，需要外部机制而非 prompting 来纠正知识。
4. **本文要解决什么？** 如何在仅有二值反馈（成功/失败）的情况下，算法化地修正 LLM 的规划知识？
5. **切入角度**：将知识纠正从"让 LLM 自己修正"转为"用算法修改外部知识库"。
6. **核心idea一句话**：算法化知识管理（用成功经验修正依赖图 + 用失败经验过滤无效动作）优于 LLM 自我纠正。

## 方法详解

### 整体框架
XENON = Adaptive Dependency Graph (ADG) + Failure-Aware Action Memory (FAM) + Context-aware Reprompting (CRe)。ADG 学习物品依赖关系，FAM 学习有效/无效动作，CRe 帮助低层控制器脱离卡住状态。

### 关键设计

1. **自适应依赖图 (ADG)**：
   - 做什么：从成功经验中修正 LLM 的错误物品依赖关系。
   - 核心算法——RevisionByAnalogy：当 agent 成功获取物品 X 时，观察其背包物品集合，与已知依赖对比，通过类比推理修正/确认依赖边。
   - 对 hallucinated 物品：RevisionByAnalogy 能通过实际经验识别不存在的物品并从图中移除。
   - 效果：400 轮后 Mineflayer 上准确率达 ~0.90。

2. **失败感知动作记忆 (FAM)**：
   - 做什么：从二值反馈中学习哪些动作有效/无效。
   - 核心思路：每个动作维护成功/失败计数，超过阈值后分类为"经验有效"或"经验无效"。
   - 无效动作在后续规划中被过滤，防止重复失败。

3. **Context-aware Reprompting (CRe)**：
   - 做什么：当控制器（如 STEVE-1）在执行中卡住时重新 prompt。
   - 检测环境状态停滞后主动中断并重新规划。

## 实验关键数据

### 长期规划成功率（学习知识 vs oracle）
| 目标类型 | Oracle 知识 SR | 学习知识 SR |
|----------|---------------|------------|
| Gold items | 0.83 | 0.74 |
| Diamond items | 0.82 | 0.64 |
| Redstone items | 0.75 | 0.28 |
| 总体 | 0.80 | 0.54 |

### 依赖学习准确率（EGA）
| 平台 | 400 轮后 |
|------|---------|
| MineRL | ~0.60 |
| Mineflayer | ~0.90 |

### 模型对比
- 7B Qwen2.5-VL + XENON > Optimus-1 (GPT-4V + oracle) 在多个目标类别上。

### 关键发现
- 准确的依赖知识是成功规划的关键——oracle 知识达 0.75 SR 的 Redstone，学习知识降至 0.00（controller 能力限制）。
- XENON 对 LLM 生成的幻觉物品具有鲁棒性（通过 RevisionByAnalogy 识别并移除）。
- LLM 自我纠正（通过 prompting）在所有基线中均失败——无法修正参数化知识错误。

## 亮点与洞察
- **"LLM 不能自我纠正参数化知识"的实证**：这一发现对 LLM Agent 设计有重要启示——不要依赖 prompt-based self-correction 来修正事实知识。
- **算法 > Prompting 的范式**：当问题的本质是知识错误而非推理错误时，算法化修正（外部记忆+统计更新）远优于自然语言反思。
- **小模型 + 好知识管理 > 大模型 + 差知识**：7B 模型 + XENON 打败 GPT-4V + oracle，说明知识管理策略比模型规模更重要。

## 局限性 / 可改进方向
- 性能受底层控制器能力限制——STEVE-1 无法执行某些复杂动作导致 Redstone 类完全失败。
- RevisionByAnalogy 有多个超参数需调优。
- 仅在 Minecraft 验证（附录有家务任务初步实验）。
- 假设依赖关系形成 DAG（无环）。

## 相关工作与启发
- **vs Optimus-1**：用 GPT-4V + oracle 依赖，XENON 用 7B + 学习依赖在多类别上更优。
- **vs Voyager/DEPS**：使用 LLM prompting 的 Minecraft agent，但不修正知识错误。

## 评分
- 新颖性: ⭐⭐⭐⭐ "算法替代自我纠正"的理念新颖且有力
- 实验充分度: ⭐⭐⭐⭐ 多平台 × 多目标类型 × 详细消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 对 LLM Agent 知识管理有重要范式启示
