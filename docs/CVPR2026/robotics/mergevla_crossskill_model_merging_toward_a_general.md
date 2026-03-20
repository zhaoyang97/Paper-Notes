# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent

**会议**: CVPR 2026  
**arXiv**: [2511.18810](https://arxiv.org/abs/2511.18810)  
**代码**: [https://mergevla.github.io/](https://mergevla.github.io/) (项目页)  
**领域**: 机器人操控 / VLA模型 / 模型合并  
**关键词**: Vision-Language-Action, 模型合并, 多任务机器人, task mask, 跨技能泛化  

## 一句话总结
MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2% 成功率。

## 背景与动机
VLA（Vision-Language-Action）模型通过微调 VLM 来做机器人操控，单任务效果不错但无法泛化到多任务。现实中需要一个通用机器人能同时掌握多个技能。直觉上可以用模型合并（model merging）把多个单任务专家合成一个，但实际上直接合并 VLA 专家会导致 **成功率降为零**——这在 LLM/VLM 合并中从未出现过，说明 VLA 有特殊的合并障碍。

## 核心问题
为什么 VLA 模型不能像 LLM/VLM 那样被成功合并？能否设计一种"为合并而生"的 VLA 架构，让多个单任务专家可以高效合并为一个通才模型？

## 方法详解
作者首先做了详细的诊断实验找到两个根因，然后针对性设计了解决方案。

### 整体框架
MergeVLA 基于 VLA-Adapter 架构（Qwen2.5-0.5B 作为 VLM backbone），但做了关键改造：(1) 在 VLM 的 LoRA 上加 task-specific binary mask 来解决参数冲突；(2) 把 action expert 的自注意力全部去掉，只保留交叉注意力；(3) 在推理时用一个免训练的 task router 自动判断当前任务。

### 关键设计
1. **Task Mask（解决 LoRA 参数冲突）**: 实验发现合并 4 个任务时，75% 以上的 LoRA 参数是"自私"的（只被一个任务的 mask 保留）。MergeVLA 对每个任务构建一个 binary mask $\mathbf{S}_m$，通过一致性检验保留与合并方向一致的参数，抑制冲突参数。公式为 $\mathbf{S}_m = \mathbb{I}[|\tau_m| > \lambda|\tau_{\text{merge}} - \tau_m|]$，其中 $\lambda$ 控制容忍度。

2. **去自注意力的 Action Expert（解决架构不兼容）**: 发现 action expert 中自注意力层在训练过程中积累了强烈的任务依赖，导致深层块参数距离爆炸式增长。解决方案很简洁：(a) 删除所有自注意力层，只保留交叉注意力——迫使 expert 依赖 VLM 的鲁棒特征；(b) 把 tanh gate 换成 sigmoid gate，避免负激活抑制 VLM 信号。浅层块直接用权重平均合并，最后一层（expert head）保持任务独立不合并。

3. **Test-time Task Router（免训练任务推断）**: 当任务身份未知时，对每个候选任务：用对应的 task mask 构建 VLM 变体 → 提取隐状态 → 投影到 action expert 的 value 子空间的主成分上 → 计算激活强度作为得分。选择得分最高的任务，只需在 t=0 时做一次路由即可。

### 损失函数 / 训练策略
各任务独立用标准模仿学习训练（30k-50k 步，batch size 8，LoRA rank 32），合并阶段完全免训练。

## 实验关键数据
| 数据集 | 指标 | MergeVLA (TIES+Mask) | 单任务微调上限 | 提升/差距 |
|--------|------|------|----------|------|
| LIBERO (4 suites) | 平均成功率 | 90.2% | 96.7% | -6.5pp |
| LIBERO-Plus (OOD) | 平均成功率 | 62.5% | 72.4% | -9.9pp |
| RoboTwin (跨具身) | 平均成功率 | 70.7% | 76.0% | -5.3pp |
| SO101 真机 (3 tasks) | 平均成功率 | 90.0% | 90.0% | 持平 |

关键对比：直接用 Task Arithmetic 合并 VLA-Adapter 成功率为 0%；MergeVLA 用同方法达到 90.2%。

### 消融实验要点
- **Mask ratio λ**: λ=0.6~0.9 效果最好（成功率>70%），太小（0.2）会导致过多冲突参数被激活
- **路由子空间选择**: 只用 Value 投影效果最好（89.7%），用 Key 或 K+V 在某些任务上直接降为 0%
- **去自注意力*的*效果**: 仅此一项修改就在 OOD 测试（LIBERO-Plus）上比 VLA-Adapter 高 13.4%
- **Expert head 深度**: 通常只需保留最后 1 层不合并；跨具身场景需要 2-3 层

## 亮点
- 诊断式研究思路非常优雅：先用实验精确定位失败的两个根因，再针对性设计解决方案
- 架构修改极简但效果显著——去掉自注意力 + 换门控函数就大幅提升泛化性
- Test-time task router 完全免训练，利用 Value 子空间的 SVD 做任务判别，很巧妙
- 在真机 SO101 上合并后性能等同单任务微调（90%），说明实用价值很高

## 局限性 / 可改进方向
- 每个任务仍需保留一个 expert head 和一个 task mask，任务数增多时存储开销线性增长
- VLM backbone 只用了 0.5B 的 Qwen2.5，是否对更大模型（7B+）同样有效尚未验证
- 当前路由只在 t=0 做一次判断，对于需要中途切换技能的长序列任务可能不够
- 跨具身实验规模较小（3 种机器人），大规模异构具身合并的可扩展性待验证

## 与相关工作的对比
- **vs OpenVLA**: OpenVLA 直接合并完全失败（0%），因为 LM body 的任务冲突无法通过简单方法解决。MergeVLA 用 task mask 绕过了这个问题
- **vs VLA-Adapter**: VLA-Adapter 的自注意力导致 action expert 不可合并，即使加了 mask 也只有在不合并最后一层时才有效。MergeVLA 从架构上消除了这个障碍
- **vs π0/π0.5**: 这些大规模 VLA 依赖联合训练实现多任务，成本高昂。MergeVLA 允许独立训练再合并，更灵活高效

## 启发与关联
- 模型合并思路对 VLM 的多任务学习也有启发——尤其是 task mask 的参数一致性检验可以迁移
- "去自注意力提升泛化性"的发现值得关注：是否在其他从头训练的模块中也存在类似现象？
- 可以探索将 MergeVLA 扩展到连续技能学习场景，每学一个新技能就合并一次

## 评分
- 新颖性: ⭐⭐⭐⭐ 诊断+设计的范式很清晰，但每个技术点（mask、去自注意力）本身不算新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个仿真 benchmark + 真机实验 + 丰富的消融和分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事逻辑非常清晰，从诊断到解决方案层层递进
- 价值: ⭐⭐⭐⭐ 解决了 VLA 合并的关键问题，对机器人多技能学习有实际意义
