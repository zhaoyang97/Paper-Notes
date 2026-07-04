---
title: >-
  [论文解读] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling
description: >-
  [NEURIPS2025][LLM Agent][text-to-3D] 提出基于图结构程序化形状表示（GPS）的多 Agent 框架 ShapeCraft，通过 Parser-Coder-Evaluator 三个 LLM Agent 协作，将自然语言分解为结构化子任务图，迭代生成可编辑、可动画的带纹理 3D 资产。
tags:
  - "NEURIPS2025"
  - "LLM Agent"
  - "text-to-3D"
  - "multi-agent"
  - "procedural modeling"
  - "shape program"
  - "Blender"
---

# ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling

**会议**: NEURIPS2025  
**arXiv**: [2510.17603](https://arxiv.org/abs/2510.17603)  
**代码**: [GitHub](https://sanbingyouyong.github.io/shapecraft)  
**领域**: LLM Agent  
**关键词**: text-to-3D, multi-agent, procedural modeling, shape program, Blender

## 一句话总结

提出基于图结构程序化形状表示（GPS）的多 Agent 框架 ShapeCraft，通过 Parser-Coder-Evaluator 三个 LLM Agent 协作，将自然语言分解为结构化子任务图，迭代生成可编辑、可动画的带纹理 3D 资产。

## 研究背景与动机

**现有问题**: 当前 text-to-3D 方法（优化方法如 SDS、自回归方法如 LLaMA-Mesh）生成的网格缺乏语义部件分割，难以编辑和动画化，不适合实际艺术工作流

**优化方法局限**: 基于 SDS 的方法需要隐式表示转显式网格（iso-surfacing），导致密集三角化、拓扑不一致等伪影

**自回归方法局限**: 直接建模三角序列的方法受训练数据分布限制，泛化能力差，生成的是整体式表示缺乏可修改性

**程序化建模的潜力**: 将形状表示为结构化程序可实现可理解、可修改的生成，但缺乏文本-程序配对数据

**LLM 直接用于 3D 的困难**: 已有工作如 3D-PREMISE 直接让 LLM 生成完整形状程序，但 LLM 对空间关系和语义形状细节的复杂文本理解能力有限，生成结果不准确

**核心洞察**: 将复杂自然语言描述分解为子任务图（GPS），可显著降低 LLM 的理解难度，配合多路径采样和视觉反馈迭代可提升生成质量

## 方法详解

### 整体框架

ShapeCraft 是一个协作式多 Agent 系统，包含三个专业 Agent（Parser、Coder、Evaluator）和一个核心共享数据结构 GPS（Graph-based Procedural Shape）。工作流程为：(1) Parser 解析输入文本构建 GPS 图；(2) Coder 为每个节点生成包围体和代码片段；(3) Evaluator 评估渲染结果并提供反馈；(4) 迭代精炼直至收敛；(5) 组件感知纹理绘制。

### 关键设计 1: GPS 图表示

GPS 表示为 $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathcal{A})$，采用扁平化的深度-1 图结构：

- **虚拟根节点** $v_0$：表示全局语义抽象（如 "chair"）
- **组件节点** $\{v_i\}_{i>0}$：每个表示一个独立几何部件，直接连接根节点
- **节点属性** $\mathcal{A}(v_i) = (n_i^g, n_i^p, b_i, p_i)$：
    - $n_i^g$：几何描述（组件的形状细节文本）
    - $n_i^p$：位置描述（空间关系与相对位置文本）
    - $b_i \in \mathbb{R}^6$：包围体参数 $(c_x, c_y, c_z, h, w, l)$
    - $p_i$：可执行的 Blender API 代码片段

**层级解析与扁平化**：Parser 从输入文本层级分解（如 chair → upper body → backrest），只保留末端节点直接连根节点，实现并行建模。

**表示引导（Representation Bootstrapping）**：通过 N=2 轮的 Evaluator 评估 → Parser+Coder 更新循环，渐进修正 GPS 中的包围体参数，缓解 LLM 幻觉问题。

### 关键设计 2: 多路径采样的迭代形状建模

对 GPS 中的每个组件节点，创建 M 条独立的建模路径并行探索：

1. **初始化**：为每个节点 $v_i$ 创建 M 个副本 $\{v_{i,m}^0\}$，Coder 根据几何描述生成初始代码
2. **迭代精炼**（T 步）：
    - Evaluator 渲染当前组件的多视角图像，生成文本反馈 $f_{i,m}^t$ 和质量分数 $s_{i,m}^t$
    - Coder 根据反馈更新代码：$v_{i,m}^{t+1} \leftarrow \text{Coder}(v_{i,m}^t, f_{i,m}^t, \mathcal{G}^*)$
3. **早停机制**：若某路径分数超过阈值 $s_\tau$，立即停止以节省计算
4. **最优选择**：选择最高分路径的结果更新 GPS

默认配置：M=3 条路径，T=3 步迭代。高温采样鼓励生成多样化的建模策略。

### 关键设计 3: 组件感知 BRDF 纹理绘制（CASD）

利用 GPS 的组件分解结构进行纹理优化：

- **纹理场** $\psi_\theta$：从 UV 坐标映射到 BRDF 参数 $(k_d, k_r, k_m)$（漫反射反照率、粗糙度、金属度），参数范围 [0,1]，可直接导入标准渲染管线
- **组件感知 SDS 损失**：

$$\mathcal{L}_{CASD} = \mathcal{L}_{SDS}(L(\psi_\theta(\mathbf{p}), \omega), x) + \sum_{i=1}^{M} \mathcal{L}_{SDS}(L(\psi_\theta(\mathbf{p}_{v_i}), \omega), n_i)$$

全局 SDS 保证整体一致性，组件级 SDS 针对各部件的几何描述 $n_i$ 提升文本对齐，仅优化各组件外部可见表面。

### 损失函数

总损失为 CASD 损失，结合全局文本引导和组件文本引导的 Score Distillation Sampling。采用 classifier-free guidance 增强文本条件控制。

## 实验关键数据

### 主实验：几何质量与文本一致性（MARVEL 子集）

| 方法 | IoGT ↑ | Hausdorff ↓ | CLIP Score ↑ | VQA Pass Rate ↑ | 运行时间 ↓ | API 调用 ↓ |
|------|--------|-------------|-------------|-----------------|-----------|-----------|
| 3D-PREMISE | 0.385 | 0.527 | 26.76 | 0.33 | 2.81 min | 6 |
| CADCodeVerify | 0.334 | 0.511 | 25.94 | 0.34 | 3.06 min | 9 |
| BlenderLLM | 0.455 | 0.511 | 26.99 | 0.43 | 5.11 min | N.A |
| LLaMA-Mesh | 0.346 | 0.464 | 25.72 | 0.28 | 15.64 min | N.A |
| MVDream | 0.427 | 0.411 | 26.84 | 0.42 | 32.10 min | N.A |
| **ShapeCraft** | **0.471** | 0.415 | **27.27** | **0.44** | 11.68 min | 21 |

ShapeCraft 在 IoGT、CLIP Score、VQA Pass Rate 三项最优，Hausdorff 距离接近最优的 MVDream，且运行时间仅为 MVDream 的 1/3。

### 消融实验：多路径采样和迭代更新

| 配置 | Hausdorff ↓ | IoGT ↑ | CLIP Score ↑ | 运行时间 ↓ |
|------|------------|--------|-------------|-----------|
| M=1, T=1 | 0.485 | 0.436 | 25.75 | 1.62 min |
| M=3, T=1 | 0.444 | 0.535 | 25.90 | 3.71 min |
| M=1, T=3 | 0.494 | 0.492 | 26.20 | 3.90 min |
| **M=3, T=3 (默认)** | **0.415** | **0.471** | **27.27** | 11.68 min |
| M=3, T=5 | 0.360 | 0.431 | 26.39 | 18.04 min |

### 与高级 thinking-mode LLM 对比（GPS 解析有效性验证）

| 方法 | IoGT ↑ | Hausdorff ↓ | CLIP ↑ | 编译成功率 ↑ |
|------|--------|-------------|--------|------------|
| ChatGPT-o3 | 0.177 | 0.708 | 25.48 | 60% |
| ChatGPT-o4-mini-high | 0.244 | 0.493 | 26.30 | 80% |
| Deepseek-R1-0528 | 0.326 | 0.489 | 29.01 | 80% |
| Gemini-2.5-Pro | 0.102 | 0.586 | 27.31 | 60% |
| **ShapeCraft** | **0.471** | **0.415** | 27.27 | **100%** |

### 关键发现

1. **GPS 显著约束 LLM 推理空间**：即使最先进的 thinking-mode LLM（o3/o4/R1/Gemini-2.5）也无法可靠生成 3D 形状程序，编译成功率仅 60-80%，而 ShapeCraft 达到 100%
2. **多路径采样比迭代更新更有效**：M=3,T=1 的 IoGT(0.535) 反而高于 M=1,T=3(0.492)，说明并行探索比单路径深挖更重要
3. **T 过大反而有害**：M=3,T=5 的 IoGT(0.431) 和 CLIP(26.39) 均低于 M=3,T=3，可能因过度修改引入退化
4. **组件感知纹理能处理复杂提示词**：如 "rust and dirt spots" 等细粒度纹理描述可被正确映射到对应组件

## 亮点与洞察

1. **GPS 表示的巧妙设计**：层级解析+扁平化存储，兼顾语义理解深度和并行建模效率；扁平结构使每个组件成为独立子任务，天然支持并行
2. **Representation Bootstrapping**：仅用 2 轮视觉反馈即可显著提升 GPS 初始质量，是一种轻量但有效的自纠正策略
3. **多路径 + 迭代的双重探索**：多路径采样增加宽度（多种建模策略），迭代更新增加深度（单策略精炼），二者互补
4. **可编辑性是核心卖点**：生成的不是死网格，而是可理解的程序 + 语义部件分割，直接支持动画和编辑
5. **组件感知纹理对齐**：利用 GPS 的组件信息将全局描述拆分为局部监督，解决了 SDS 对复杂提示词的弱对齐问题

## 局限性

1. **提示词质量敏感**：歧义、过短或创意性提示词仍会导致 Parser 分解不准确、Evaluator 信号不足
2. **复杂有机形状困难**：如尾巴、翅膀等有机几何体受限于 Coder 的 Blender API 库范围
3. **运行时间较长**：21 次 API 调用、11.68 分钟运行时间，相比直接方法（3D-PREMISE 2.81 min）慢 4 倍
4. **依赖特定 LLM**：使用 Qwen3-235B 作为 Parser/Coder、Qwen-VL-Max 作为 Evaluator，换用其他模型的泛化性未验证

## 相关工作与启发

- **vs 3D-PREMISE / CADCodeVerify**：ShapeCraft 通过 GPS 约束推理空间，避免了直接生成完整程序的失败；编译成功率从 60-80% 提升到 100%
- **vs MVDream（优化方法）**：ShapeCraft 生成结构化可编辑网格，MVDream 生成稠密不可编辑网格；ShapeCraft 运行速度快 3 倍
- **vs 3D-GPT**：3D-GPT 只做场景级资产检索与布局，不涉及细粒度形状建模；ShapeCraft 解决的是单物体的精细形状生成
- **启发**：multi-agent + 结构化中间表示的范式可推广到其他生成任务（如代码生成、文档写作），GPS 的"层级分析 + 扁平执行"思路值得借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ (GPS 表示和组件感知纹理是新颖贡献，多 Agent 框架本身不新但与 3D 结合有创新)
- 实验充分度: ⭐⭐⭐⭐ (涵盖定性定量对比、多维度消融、与 thinking-mode LLM 对比，缺少用户研究)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图示丰富，算法伪代码规范)
- 价值: ⭐⭐⭐⭐ (首个在 text-to-3D 中实现 100% 编译率的 LLM Agent 方案，实用性强)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[NeurIPS 2025\] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [\[ACL 2025\] BookWorld: From Novels to Interactive Agent Societies for Story Creation](../../ACL2025/llm_agent/bookworld_from_novels_to_interactive_agent_societies_for_story_creation.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](../../ACL2026/llm_agent/codestruct_code_agents_over_structured_action_spaces.md)

</div>

<!-- RELATED:END -->
