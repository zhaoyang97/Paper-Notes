# Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling

**会议**: NeurIPS 2025  
**arXiv**: [2504.13169](https://arxiv.org/abs/2504.13169)  
**代码**: 有（项目页面/GitHub/HuggingFace）  
**领域**: 多模态VLM / AI安全  
**关键词**: VLM幻觉, 自验证, 回溯纠正, 置信度token, 生成式自校正  

## 一句话总结
提出REVERSE框架——首次在单一VLM内统一了生成、验证和纠正三个阶段：通过引入<SPAN>、</CN>（置信）、</UN>（不置信）三个特殊token训练幻觉感知模型，推理时当</UN>概率超过阈值就回溯到上一个</CN>重新生成，在CHAIR-MSCOCO上降低12%、HaloQuest上降低34%的幻觉率。

## 背景与动机
VLM幻觉缓解方法分两大类：(1) 生成调整（VCD/OPERA/DoLA等），修改解码行为但无法纠正已生成的错误token；(2) 事后验证（Woodpecker/LURE），用外部模型检查和改写但需要复杂多模型pipeline。两者的根本局限：生成类无法纠错，验证类无法自我纠错。

## 核心问题
如何让VLM在生成过程中实时验证自己是否在幻觉，并即时回溯纠正——不需要外部模型？

## 方法详解

### 整体框架
训练+推理两阶段：训练阶段用1.3M半合成数据教VLM标记哪些短语可信/不可信；推理阶段实时监控</UN>概率，超阈值则回溯+重采样纠正。

### 关键设计

1. **三个置信度特殊token**: 
   - `<SPAN>`: 标记关键短语开始
   - `</CN>`: 标记置信（confident）短语结束——这是"存档点"
   - `</UN>`: 标记不置信（unconfident/hallucinated）短语结束——触发回溯
   正确短语标记为`<SPAN>a young girl</CN>`，幻觉短语标记为`<SPAN>a frisbee</UN>`

2. **1.3M半合成训练数据**: 从LLaVA-v1.5-665k出发，用规则+GPT-4o-mini生成负样本（错误答案/幻觉描述）。3.8M正确QA对+2.9M幻觉QA对，总计6.8M轮。负样本在</UN>后立即终止（防止继续生成无意义文本）。

3. **回溯式重采样（Retrospective Resampling）**: 推理时：
   - 正常生成，监控每个token后</UN>出现的概率
   - 当P(</UN>) > τ（阈值），触发回溯到最近的</CN>位置
   - 用两种策略纠正：(a) 提高温度做拒绝采样；(b) 将潜在幻觉短语作为Hint注入query重写
   - 可迭代——如果纠正后仍然幻觉，继续回溯

4. **幻觉感知训练损失**: 修改交叉熵损失，同时训练三个目标：正常文本生成+短语置信度分类+幻觉token概率抑制。

## 实验关键数据

**CHAIR-MSCOCO (LLaVA-v1.5-7B)**:

| 方法 | CHAIRi↓ | CHAIRs↓ |
|------|---------|---------|
| 无处理 | 15.4 | 50.0 |
| VCD | 14.9 | 48.6 |
| OPERA | 14.6 | 47.8 |
| HA-DPO | 11.0 | 38.2 |
| EOS | 12.3 | 40.2 |
| Woodpecker | 14.8 | 45.8 |
| **REVERSE(τ=0.003)** | **10.3** | **37.0** |
| **REVERSE(τ=0.0003)** | **6.1** | **13.6** |

- τ=0.0003时CHAIRs从50.0降到13.6——幻觉率降低超72%！
- 在LLaVA-MORE和Qwen2.5-VL上也有效
- HaloQuest上提升34%

### 消融实验要点
- **τ阈值**: 较低的τ更激进地检测幻觉，CHAIRs极低但可能过度拒绝（Cover下降）
- **拒绝采样 vs query重写**: 两者都有效，query重写对complex hallucination更好
- **迭代次数**: 1-2次回溯最优，过多迭代导致output过短
- **训练数据**: 同时包含正负样本比只用正样本显著更好
- **通用benchmark**: 在MMBench/POPE等通用任务上性能保持或微升

## 亮点
- **首次统一生成-验证-纠正**: 不需要外部模型，VLM自己就是generator+verifier+corrector
- **优雅的token设计**: `</CN>`作为"存档点"、`</UN>`作为"幻觉警报"——概念直观、实现简洁
- **回溯机制**: 类似游戏存档/读档——检测到问题就"回到上一个存档点重来"
- **可控的精度-召回权衡**: 通过τ阈值灵活控制幻觉检测的激进程度

## 局限性 / 可改进方向
- 低τ时虽然幻觉大幅减少，但Cover（覆盖率）也下降——模型倾向于"少说少错"
- 回溯+重采样增加推理延迟（每次回溯需要部分重新生成）
- </UN>概率的校准依赖训练数据分布，换模型或域外可能失灵
- 1.3M训练数据的构建依赖GPT-4o-mini，有数据偏差风险
- 仅在captioning和简单QA验证，复杂推理任务的效果未知

## 与相关工作的对比
- **vs VCD/OPERA/DoLA（生成调整）**: REVERSE在检测幻觉后可以纠正，它们不能
- **vs Woodpecker（事后验证）**: REVERSE自己做验证不需要外部GPT-4，且可以即时纠正而非事后改写
- **vs VHR（注意力头增强）**: VHR在attention层面增强视觉，REVERSE在output层面检测和纠正——互补
- **vs Visual Evidence Prompting**: VEP添加外部视觉信息预防幻觉，REVERSE检测并纠正幻觉——也互补

## 启发与关联
- 置信度token的设计可以推广到Agent——Agent执行操作时标记</CN>和</UN>，不确定时回溯
- REVERSE的回溯机制与VReST的MCTS互补：VReST在推理空间广度探索，REVERSE在生成过程中纵深纠错
- TVC（视觉遗忘缓解）+REVERSE可能形成强力组合：TVC在推理中重注入视觉信息+REVERSE在生成中检测纠错

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 统一生成-验证-纠正是全新范式，置信度token和回溯机制设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 3个VLM backbone、6个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1和3的示意图极其清晰
- 价值: ⭐⭐⭐⭐⭐ 对VLM幻觉缓解方向有里程碑意义
