---
title: >-
  [论文解读] OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning
description: >-
  [ICLR 2026][机器人][视觉语言动作模型] OneTwoVLA 把机器人里的快动作执行和慢语言推理统一到同一个 VLA 中，让模型在关键时刻用 `[BOR]` 触发推理、平时用 `[BOA]` 直接输出动作，从而在长程操作、错误恢复、人机交互和开放视觉 grounding 上明显优于无推理 VLA 与双系统方案。
tags:
  - "ICLR 2026"
  - "机器人"
  - "视觉语言动作模型"
  - "自适应推理"
  - "长程操作"
  - "人机交互"
  - "视觉 grounding"
---

# OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=tWMfhoP3as](https://openreview.net/forum?id=tWMfhoP3as)  
**代码**: 暂未公开  
**领域**: 机器人 / 具身智能  
**关键词**: 视觉语言动作模型, 自适应推理, 长程操作, 人机交互, 视觉 grounding

## 一句话总结
OneTwoVLA 把机器人里的快动作执行和慢语言推理统一到同一个 VLA 中，让模型在关键时刻用 `[BOR]` 触发推理、平时用 `[BOA]` 直接输出动作，从而在长程操作、错误恢复、人机交互和开放视觉 grounding 上明显优于无推理 VLA 与双系统方案。

## 研究背景与动机

**领域现状**：通用机器人控制最近主要沿着 vision-language-action model (VLA) 发展：模型看多视角图像和语言指令，直接输出连续动作或动作 token。另一条路线借鉴人类的双系统认知，把高层 VLM / LLM 当作慢速 System Two，让它先规划或分解任务，再由低层 VLA / policy 作为 System One 执行动作。

**现有痛点**：纯 VLA 执行动作很快，但面对长程任务时容易忘记当前进度，也难以处理错误、歧义和人类临场改指令。双系统方法能显式推理，却带来两个很实际的问题：高层模型不知道低层 policy 到底会不会做某个动作，可能给出不可执行的子任务；同时高层 VLM 推理延迟很高，如果固定频率反复调用，机器人会等得慢，甚至拿到已经过时的指导。

**核心矛盾**：机器人需要同时具备“快”和“慢”两种能力。快动作执行要求低延迟、闭环、贴近传感器反馈；慢语言推理要求看懂场景、维护计划、追踪历史、处理异常。如果把两者硬拆成两个模型，协同成本高；如果把推理完全去掉，长程泛化又不够。

**本文目标**：作者希望得到一个单模型 policy，它既能在普通时刻像 System One 一样连续执行动作，又能在完成子任务、检测到错误、遇到人类干预或歧义时像 System Two 一样输出自然语言推理，并且让这个切换由模型自己决定。

**切入角度**：关键观察是，推理不需要每一帧都发生。多数时间机器人只要根据最近一次推理继续执行动作；只有当状态发生语义层面的变化，才需要重新描述场景、更新计划和决定下一步。因此，与其“每步都推理”或“从不推理”，不如让同一个 VLA 学会判断什么时候推理。

**核心 idea**：用一个统一的 VLA 同时建模决策 token、语言推理和动作 chunk，并用带具身推理标注的机器人数据与合成视觉语言数据共同训练，让模型在执行过程中自适应地在 Reason 和 Act 两种模式之间切换。

## 方法详解

### 整体框架

OneTwoVLA 的输入包括当前多相机观测 $I_t^{1:n}$、最近一次推理时刻的参考图像 $I_{ref}^{1:n}$、语言指令 $\ell$、最新推理内容 $R$，以及在动作模式下额外使用的机器人 proprioceptive state $s_t$。模型每个时间步先输出一个 decision token：如果是 `[BOR]`，它进入 reasoning mode，生成新的文本推理；如果是 `[BOA]`，它进入 acting mode，基于最近推理直接生成动作 chunk $A_t$ 并执行。

整体上，它不是把 VLM planner 和 VLA controller 拼起来，而是在同一个模型内部把“何时想一想”和“何时动手做”合成一个闭环。推理内容会保存为状态，动作执行期间持续被复用；当模型判断推理过期、任务阶段变化、动作失败或人类输入改变目标时，再刷新推理。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A["当前观测 + 参考图像<br/>指令 + 最新推理"] --> B["自适应推理/动作切换"]
	B -->|[BOR]| C["具身推理状态<br/>场景/计划/历史/下一步"]
	C --> A
	B -->|[BOA]| D["统一 VLA 动作生成<br/>输出动作 chunk"]
	D --> E["机器人执行<br/>更新环境状态"]
	E --> A
	F["推理增强数据构建"] --> B
	G["合成具身 VL 数据<br/>提升泛化"] --> B
```

### 关键设计

**1. 自适应推理/动作切换：用 `[BOR]` 和 `[BOA]` 把慢思考变成可学习的控制决策**

传统双系统机器人通常把高层推理的调用频率写死，比如每隔一段时间让 VLM 重新规划。OneTwoVLA 改成让模型先预测 decision token。给定当前图像、参考图像、指令和最近推理 $R$，如果模型认为需要更新语义状态，就输出 beginning of reasoning `[BOR]`，随后自回归生成新的推理文本；如果最近推理仍然足够新，就输出 beginning of action `[BOA]`，直接进入动作专家生成动作。

这个设计的核心不是多加两个特殊 token，而是把“推理是否过期”变成 policy 自己学习的中间决策。对于番茄炒蛋、火锅、鸡尾酒这类任务，机器人只有在完成一个食材步骤、抓取失败、人类打断或目标模糊时才需要重新推理；其他时刻反复生成语言只会拖慢执行。论文的时间分析也支持这一点：OneTwoVLA 的总任务时间接近不推理的 flat VLA，而双系统“总在推理”方案会因为 Gemini 2.5 Pro 的调用延迟明显变慢。

**2. 具身推理状态：让语言推理同时承担场景理解、计划、进度记忆和下一步控制接口**

OneTwoVLA 生成的 reasoning content 不是一段自由发挥的 chain-of-thought，而是结构化的具身状态，包含四类信息：场景描述、全局高层计划、历史摘要、当前下一步。场景描述聚焦任务相关物体的位置；计划列出完成任务的步骤；历史摘要告诉模型已经完成了什么；下一步把长程目标压缩成当前要执行的动作意图。

这个结构解决了 VLA 在长程任务中最容易出错的两个问题。第一，模型不必只靠视觉帧隐式记忆进度，而是能读到“我已经加了糖浆和橙汁，现在要加伏特加”这样的历史状态。第二，动作专家收到的不只是原始用户指令，而是结合当前场景和进度后的下一步意图，因此更容易学习精细动作。论文还加入最近推理时刻的参考图像 $I_{ref}^{1:n}$，用于给当前图像提供历史锚点，避免相似状态下模型分不清“刚开始”还是“已经完成一步”。

**3. 推理增强机器人数据构建：把人工示范切成推理区间和动作区间来监督切换行为**

现有机器人数据大多只有 observation-action pair，不告诉模型什么时候该停下来重新理解任务。作者因此把每条人工示范轨迹切成两类 interval：reasoning interval 出现在子任务完成、错误出现、需要人机交互等关键点，并标注推理文本；acting interval 则主要学习根据最新推理继续执行动作。训练时，reasoning interval 中如果当前 $R$ 已经过期，监督 decision token 为 `[BOR]`；当 $R$ 更新后，监督变成 `[BOA]`；acting interval 中始终监督 `[BOA]`。

推理标注本身采用两阶段自动 pipeline。第一阶段先给定高层计划 $P=(p_1,\dots,p_K)$，从示范视频均匀采样 $N=32$ 帧，让 Gemini 2.5 标出每个子任务完成后的 reasoning interval。第二阶段在每个 interval 的中点帧上生成四个字段：场景描述 $D_j$、完整计划 $P$、历史摘要 $H_j=(p_1,\dots,p_j)$、下一步 $X_j=p_{j+1}$。作者报告在 Tomato-Egg 任务中，人类评估认为 81.5% 的 interval 标注正确，83.3% 的场景描述合理，说明这个自动标注虽然不是完美真值，但质量足以支撑训练。

**4. 合成具身视觉语言数据：用 action-free VL 样本补足泛化推理能力**

机器人示范昂贵，光靠真实轨迹很难覆盖开放物体、空间关系、属性描述和用户意图。OneTwoVLA 利用统一模型的好处，把没有动作标签的视觉语言数据也纳入共同训练。作者先让 Gemini 2.5 Pro 生成多样化桌面场景文字描述，再用 FLUX.1-dev 合成图像，并加入鱼眼畸变或机器人夹爪合成，让图像更接近真实机器人视角；最后再让 Gemini 生成任务指令与对应推理内容。

这条 pipeline 一共生成 16,000 个样本，其中 6,000 个偏视觉 grounding，每张图含直接名称、空间关系、属性和语义特征等指代表达；另外 10,000 个偏长程计划，要求从目标生成多步计划，有一部分还加入人机交互。它没有动作标签，所以不能直接教机器人怎么动，但能激活预训练 VLM 中的常识和视觉语义能力，再通过与机器人数据 co-training 迁移到控制 policy 中。实验里 OneTwoVLA-VL 在开放世界视觉 grounding 和泛化规划上大幅超过只用机器人数据的 OneTwoVLA，正说明这部分数据不是装饰，而是泛化来源。

### 一个完整示例

以“Make me a cup of Vodka Sunrise”为例，初始观测中桌上有糖浆、橙汁、不同口味伏特加和杯子。模型先输出 `[BOR]`，生成推理：场景中糖浆在某处、橙汁在某处；计划是先加糖浆、再加橙汁、最后加伏特加；历史为空；现在要加糖浆。随后多帧内它输出 `[BOA]`，动作专家连续生成抓瓶、移动、倾倒、放回等动作 chunk。

当糖浆加入完成后，当前视觉状态和参考图像、历史推理不再一致，模型再次输出 `[BOR]`，把历史摘要更新为“已加入糖浆”，下一步变成“加入橙汁”。如果这时人类打断说“不想要橙味伏特加，换柠檬味”，交互文本会追加进语言指令，模型再进入 reasoning mode，把计划改成“放下橙味伏特加，拿柠檬味伏特加”。这就是统一模型比双系统更顺的地方：同一个模型既知道当前低层执行到了哪里，也能立刻把语言交互转成可执行的下一步。

### 损失函数 / 训练策略

OneTwoVLA 以 $\pi_0$ 作为基础 VLA 实例。其 vision-language model 部分负责自回归生成文本推理，并用交叉熵监督 decision token 和 reasoning token；连续动作分布沿用 $\pi_0$ 的 action expert，用 flow matching loss 训练动作 chunk。形式上，reasoning mode 生成 $\hat{R}\sim\pi_\theta(\cdot|I_t^{1:n}, I_{ref}^{1:n}, \ell, R)$，acting mode 生成 $A_t\sim\pi_\theta(\cdot|I_t^{1:n}, I_{ref}^{1:n}, \ell, R, s_t)$。

训练数据由三部分混合：普通机器人示范动作数据、带 reasoning interval 的推理增强机器人数据、以及 16,000 个 action-free 合成具身 VL 样本。前两者教模型何时推理、何时动作以及动作如何执行；后者主要监督语言推理和视觉语义理解，用于补足未见物体、未见指令和常识规划。

## 实验关键数据

### 主实验

论文用真实机器人实验验证四种能力：长程任务规划、错误检测与恢复、自然人机交互、视觉 grounding。长程任务包括 Tomato-Egg、Hotpot 和 Cocktail，每个任务每个方法评估 20 次；主要对比 flat $\pi_0$、双系统 VLA 和 OneTwoVLA。

| 任务 / 设置 | 指标 | OneTwoVLA | 之前 SOTA / 基线 | 提升 |
|--------|------|------|----------|------|
| 长程任务平均 | 成功率 | 87% | $\pi_0$: 57% | +30 个百分点 |
| 长程任务平均 | 成功率 | 87% | Dual-System: 63% | +24 个百分点 |
| Tomato-Egg | 成功率 | 85% | $\pi_0$: 70%, Dual-System: 55% | +15 / +30 个百分点 |
| Hotpot | 成功率 | 80% | $\pi_0$: 50%, Dual-System: 70% | +30 / +10 个百分点 |
| Cocktail | 成功率 | 95% | $\pi_0$: 50%, Dual-System: 65% | +45 / +30 个百分点 |

从这个表看，OneTwoVLA 的优势不是单一任务偶然爆发，而是在三个长程真实操作上都稳定更高。$\pi_0$ 的典型失败是忘记进度或重复当前步骤，例如火锅里反复夹牛肉；双系统的失败则更像接口错配和延迟，比如高层系统给出低层 policy 不会执行的命令，或者推理来得太晚。

### 消融实验

论文没有做传统“去掉某个 loss”的内部消融，但给出了非常清楚的组件对比：无推理的 $\pi_0$、统一但只用机器人数据的 OneTwoVLA、以及加入 16,000 合成 VL 数据的 OneTwoVLA-VL。这个对比能看出显式推理和 VL co-training 分别贡献了什么。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $\pi_0$ flat VLA | 长程平均 57%；Single-Env grounding 5%；Open-World grounding 3% | 动作执行快，但缺少显式推理和历史状态，长程计划与语义 grounding 都弱 |
| Dual-System VLA | 长程平均 63%；人机交互 65%；错误恢复 58.3% | 有高层推理，但高低层互不了解，且 VLM 延迟导致更新不及时 |
| OneTwoVLA | 长程平均 87%；错误恢复 80%；人机交互 100%；Single-Env grounding 78% | 统一模型 + 自适应推理，显著提升长程执行、恢复和交互 |
| OneTwoVLA-VL | 泛化规划平均 70%；Open-World grounding 73%；泛化人机交互 72.5% | 加入合成具身 VL 数据后，未见任务、未见物体和复杂指代表达泛化大幅提升 |

更细的视觉 grounding 分解也很有意思。在 Single-Env 中，OneTwoVLA-VL 的 name / spatial / attribute / semantic 成功数分别是 10/10、8/10、8/10、9/10；OneTwoVLA 是 10/10、5/10、8/10、8/10；$\pi_0$ 只有 2/10、0/10、0/10、0/10。在 Open-World 中，OneTwoVLA-VL 达到 29/40，而 OneTwoVLA 只有 3/40，$\pi_0$ 只有 1/40，说明合成 VL 数据主要改善开放环境泛化，而不只是让已见环境上的语义描述更顺。

### 关键发现

- 自适应推理确实能同时兼顾效率和能力。OneTwoVLA 只在关键时刻推理，任务完成时间接近不推理的 $\pi_0$；双系统因为频繁调用 Gemini 2.5 Pro，推理时间占比高，真实执行慢很多。
- 显式 reasoning content 不只是给人看的解释，也帮助动作学习。作者报告 $\pi_0$ 在验证集上的动作 MSE 比 OneTwoVLA 高 62%，说明“下一步 + 历史状态”确实让动作预测更容易。
- 合成 VL 数据主要解决机器人数据覆盖不到的开放语义问题。OneTwoVLA 不加 VL co-training 时，在 Open-World grounding 只有 8% 成功率；加上后到 73%，还能处理 Sprite、GoPro 等不在机器人数据甚至不在合成数据中的对象。
- 错误恢复和人机交互是统一模型的强项。错误恢复中 OneTwoVLA 达到 8/10，双系统和 $\pi_0$ 分别是 7/12、8/14；人机交互中 OneTwoVLA 在 Hotpot 和 Cocktail 的 20 次交互全部成功，双系统只有 13/20。

## 亮点与洞察

- 把“什么时候推理”显式建模成 token 决策很巧。很多 embodied reasoning 方法把推理当成固定流程，OneTwoVLA 则让推理频率随状态变化而变化，这更接近真实机器人部署时的效率约束。
- 统一模型比双系统的关键优势在于能力边界共享。高层推理和低层动作来自同一个模型、同一套训练数据和同一段历史状态，减少了 planner 说得出但 controller 做不到的问题。
- reasoning content 的四字段设计很实用。场景、计划、历史、下一步刚好覆盖长程任务最容易丢失的信息，而且格式足够简单，可以用自动标注 pipeline 大规模生成。
- 合成视觉语言数据的价值被放在了机器人语境里。它不是普通 VQA 数据，而是桌面场景、物体指代、长程计划和人机交互，所以和机器人 policy 的输入分布更接近。
- 这篇对 VLA scaling 有一个启发：如果 VLA 初始化自强 VLM，那么 action-free 数据并非只能做预训练，也可以在 policy co-training 阶段继续激活模型的语义常识。

## 局限与展望

- 作者承认目前 reasoning interval 依赖人工设计的候选关键步骤和自动标注，相当于一种 SFT 式对齐；什么时候推理未必最优。未来可以用 RL 让模型根据任务成功率、延迟和错误代价学习更优推理策略。
- 即使只在少数关键时刻推理，机器人仍要停顿 2-3 秒。对于需要连续动态控制的任务，异步推理和动作生成会更重要，例如边执行边预估下一段计划。
- 动作推理效率还没有被充分优化。随着统一模型变大，action expert 或 VLM backbone 的推理成本可能成为瓶颈，需要引入更快的 action decoding、缓存或蒸馏。
- 合成 VL 数据虽然规模化，但质量仍有噪声。作者抽样 50 个合成样本发现 2 个图像错误、8 个文本错误、40 个完全正确；这说明数据有用，但如果继续扩大规模，质量过滤会变得更关键。
- 论文主要在桌面操作与厨房/饮品/物体拾取场景验证，还没有证明对移动操作、多机器人协作或更长时间尺度任务同样有效。

## 相关工作与启发

- **vs $\pi_0$ / OpenVLA 类 flat VLA**: 这些方法直接从视觉语言输入到动作输出，部署简单、低延迟，但缺少显式进度追踪和可交互推理。OneTwoVLA 沿用 $\pi_0$ 的 action expert 能力，同时加上自适应语言推理，牺牲很小延迟换来长程和开放语义能力。
- **vs Hi Robot / ViLa 双系统框架**: 双系统把高层 VLM 和低层 VLA 明确分开，结构直观，但接口和延迟是硬伤。OneTwoVLA 的统一模型让推理内容直接服务于动作生成，也让模型能根据执行状态决定是否重新推理。
- **vs $\pi_0.5$**: $\pi_0.5$ 也是单模型开放泛化方向，但它倾向于每步预测较简单、信息有限的子任务。OneTwoVLA 的推理内容更完整，包含场景、计划、历史和下一步，并且不是每一步都强制生成。
- **vs ECoT-Lite / embodied chain-of-thought 压缩路线**: 避免测试时推理可以降低延迟，但会削弱人机交互和在线纠错。OneTwoVLA 的折中是只在关键时刻显式推理，让推理仍然参与闭环。
- **对后续工作的启发**: 可以把 `[BOR]/[BOA]` 扩展成更细粒度的模式选择，例如快速动作、短文本更新、完整重规划、询问人类、调用外部工具等，让 VLA 的执行循环更像一个可学习的实时调度器。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 用统一 VLA 学自适应推理/动作切换，抓住了双系统机器人最痛的延迟和接口错配问题。
- 实验充分度: ⭐⭐⭐⭐⭐ 真实机器人实验覆盖长程规划、错误恢复、人机交互、视觉 grounding 和开放泛化，指标很完整。
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，方法、数据 pipeline 和实验任务都解释得比较具体，案例也直观。
- 价值: ⭐⭐⭐⭐⭐ 对具身智能很有参考价值，尤其是“推理不必每步发生，但必须能在线发生”这个设计原则很可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] UniVLA: Unified Vision-Language-Action Model](unified_vision-language-action_model.md)
- [\[ICLR 2026\] Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning](vlaser_vision-language-action_model_with_synergistic_embodied_reasoning.md)
- [\[ICLR 2026\] HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](hybridvla_collaborative_diffusion_and_autoregression_in_a_unified_vision-languag.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/robotics/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process](unified_diffusion_vla_vision-language-action_model_via_joint_discrete_denosing_d.md)

</div>

<!-- RELATED:END -->
