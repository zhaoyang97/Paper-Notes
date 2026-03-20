# CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance

**会议**: ICCV 2025  
**arXiv**: [2412.20451](https://arxiv.org/abs/2412.20451)  
**代码**: [https://chain-of-affordance.github.io](https://chain-of-affordance.github.io)  
**领域**: 多模态VLM / 具身智能 / 机器人操作  
**关键词**: VLA, chain-of-affordance, robot manipulation, visual prompting, affordance reasoning  

## 一句话总结
提出Chain-of-Affordance（CoA-VLA）框架，将四类机器人affordance（物体、抓取、空间、运动）以文本和视觉双模态形式注入VLA模型的策略网络，在真实机器人7任务多任务学习中达到85.54%成功率，比OpenVLA高30.65%，并展现出对未见物体姿态和障碍物的泛化能力。

## 研究背景与动机
1. **领域现状**：VLA模型通过大规模预训练获得了强大的泛化能力，但现有方法要么依赖LLM/VLM做高层规划（外部推理），要么端到端直接预测动作（缺乏推理）。OpenAI O1展示了长链推理可以显著提升复杂问题解决能力。
2. **现有痛点**：当前VLA模型在复杂环境中缺乏自驱动的中间推理能力，导致在需要精确抓取、空间推理和避障的任务中容易失败。已有推理方法如ECoT侧重任务分解，但缺乏对物理交互的结构化理解。
3. **核心矛盾**：机器人执行复杂操作需要理解物体在哪、怎么抓、放哪里、怎么移动这一连串问题，但现有VLA没有显式地建模这些中间推理步骤。
4. **本文要解决什么**：设计一种结构化的affordance推理链，让VLA模型在预测动作前先推理出与任务相关的四类affordance，并将结果注入策略网络。
5. **切入角度**：从机器人affordance的经典概念出发，将其形式化为CoT推理链，并创新性地用文本+视觉双模态表示。
6. **核心idea一句话**：用四类affordance（object/grasp/spatial/movement）构建推理链，以文本和视觉双格式注入VLA的diffusion策略头来指导动作生成。

## 方法详解

### 整体框架
基于DiffusionVLA（Qwen2-VL + diffusion action head），输入是机器人观测图像和任务指令，模型先自回归地生成affordance推理链（文本形式），同时生成对应的视觉affordance标注叠加在观测图上。两种模态的affordance通过co-injection模块融合后注入diffusion策略网络生成连续动作。

### 关键设计

1. **四类Affordance定义**:
   - **Object affordance**：识别目标物体及其在图像中的位置（用bounding box表示），解决"操作什么、在哪里"的问题。
   - **Grasp affordance**：确定物体上最适合抓取的部位（用2D关键点表示），如茶壶的壶柄。
   - **Spatial affordance**：识别环境中的可用空间（如盘子上的空白区域），用于确定放置位置。
   - **Movement affordance**：规划避碰运动轨迹，确保机器人安全移动。
   - 四类affordance构成一个序列链：先知道操作什么→怎么抓→放哪里→怎么移动过去。这种链式结构让推理有明确的顺序依赖关系。

2. **视觉-文本双模态Affordance表示**:
   - **文本affordance**：用自然语言描述每类affordance及其坐标信息（如bounding box坐标对、关键点坐标）。使用ChatGPT对affordance描述做语言多样化增强，避免模板化偏差。
   - **视觉affordance**：将affordance信息直接叠加在观测图像上——bounding box、抓取点用高对比度标注，运动轨迹用低显著度细线标注。这种层次化视觉编码让模型一眼就能区分不同affordance。
   - 设计动机：文本affordance提供语义丰富的推理信号，视觉affordance提供像素对齐的空间感知。两者互补，单独使用效果都不如联合使用好。

3. **视觉-文本Co-Injection模块**:
   - 做什么：将文本和视觉affordance统一融合后注入diffusion策略网络。
   - 核心思路：文本affordance通过VLM最后一层embedding + MLP投射为token序列；视觉affordance通过预训练ViT-Small抽取patch token。两组token经过2层Transformer block进行跨模态融合，最终通过FiLM conditioning层注入diffusion model。
   - 设计动机：FiLM层可以动态调制diffusion过程，让策略生成的动作同时考虑空间约束和语义意图，而不需要改变diffusion框架的整体结构。

4. **动态Affordance选择机制**:
   - 做什么：根据任务进度和机器人状态自适应选择需要哪些affordance，避免全部计算。
   - 核心思路：将本体感知信息（关节角度等）编码为单个token拼接到视觉token前面，让模型学习在不同时间步智能选择相关affordance。例如：夹爪闭合+腕部摄像头看到物体→跳过object和grasp affordance，只生成spatial和movement affordance。
   - 效果：推理速度6Hz（vs 无动态选择时1Hz），且去掉动态选择后精度反而下降（冗余affordance引入噪声）。

### 数据生成Pipeline
使用GPT-4o生成场景描述和实体识别 → Grounding DINOv2 + SAM生成bounding box → RoboPoint + GPT-4o预测空间affordance点并聚类 → CoTracker追踪机器人末端轨迹获取运动affordance。整个pipeline自动化，大幅减少人工标注需求。

## 实验关键数据

### 主实验（真实机器人，7任务多任务学习）

| 模型 | In-Distribution Avg | Visual Generalization Avg |
|------|-------------------|--------------------------|
| Diffusion Policy | 33/77 (42.93%) | 3/63 (4.76%) |
| Octo | 34/77 (44.13%) | 12/63 (19.05%) |
| OpenVLA | 52/77 (54.89%) | 14/63 (22.22%) |
| DiffusionVLA | 59/77 (76.60%) | 28/63 (44.44%) |
| **CoA-VLA** | **64/77 (85.54%)** | **36/63 (57.14%)** |

LIBERO仿真benchmark：

| 模型 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.6 | 76.5 |
| **CoA-VLA** | **88.0** | **90.4** | **82.0** | **59.0** | **79.8** |

### 消融实验

| 配置 | LIBERO Avg | 说明 |
|------|-----------|------|
| CoA-VLA (Full) | 79.8 | 完整模型 |
| w/o visual affordance | 降低 | 去掉视觉affordance |
| w/o textual affordance | 降低更多 | 文本affordance贡献更大 |
| w/o dynamic selection | 降低 + 慢6x | 冗余affordance引入噪声 |

### 关键发现
- 文本affordance比视觉affordance贡献更大，因为语言能编码更丰富的任务语义。
- 不使用动态选择反而更差，说明冗余affordance是噪声而非信息增益。
- CoA-VLA在视觉泛化场景优势更明显（57.14% vs DiffusionVLA 44.44%），说明affordance推理显著增强了模型对视觉变化的鲁棒性。
- 空间affordance使机器人能识别盘子上的空白区域准确放置物品，运动affordance使机器人能绕开障碍物。
- 模型能泛化到未见物体姿态（如茶壶把手朝不同方向），但完全水平放置时仍失败。

## 亮点与洞察
- **Affordance作为结构化CoT**：将经典的affordance概念与现代CoT推理结合是一个很自然但巧妙的桥接。affordance本身就是"物体能被怎么操作"的回答，天然适合作为机器人动作预测前的中间推理。
- **视觉-文本双模态注入**：不只是说出affordance（文本），还画出来（视觉叠加），两种模态通过FiLM注入diffusion head。这种设计可以迁移到任何diffusion-based VLA。
- **动态选择减少冗余**：不是越多信息越好，冗余affordance反而是噪声。利用本体感知做简单的选择就能6x加速且不掉点。

## 局限性 / 可改进方向
- 依赖GPT-4o、Grounding DINO、SAM、CoTracker等多个外部模型生成affordance数据，pipeline较复杂。
- 只在Franka单臂7任务+LIBERO上验证，任务多样性和规模有限。
- 动态affordance选择基于本体感知的简单判断，更复杂的策略（如基于注意力的选择）可能更好。
- 物体完全水平放置时抓取失败，说明affordance预测在极端姿态下仍有困难。

## 相关工作与启发
- **vs ECoT**: ECoT用任务分解+子任务描述+运动指令做推理，本文用四类affordance做推理，更贴近物理交互的本质。
- **vs CoT-VLA**: CoT-VLA生成sub-goal来指导自回归VLA，本文生成的是更细粒度的affordance信息并注入diffusion head。
- **vs TraceVLA**: TraceVLA用视觉轨迹作为额外输入，本文将视觉affordance不仅作为输入还作为推理的中间产物。

## 评分
- 新颖性: ⭐⭐⭐⭐ 四类affordance + 双模态表示 + co-injection是比较新颖的组合，但单个idea不算全新
- 实验充分度: ⭐⭐⭐⭐ 真实机器人7任务 + LIBERO仿真，有泛化实验和消融，比较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，affordance定义和方法描述详细
- 价值: ⭐⭐⭐⭐ 为VLA引入结构化推理提供了一个实用的方向，对具身智能社区有启发
