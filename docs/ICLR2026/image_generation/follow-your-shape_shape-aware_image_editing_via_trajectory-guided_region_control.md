# Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control

**会议**: ICLR 2026  
**arXiv**: [2508.08134](https://arxiv.org/abs/2508.08134)  
**代码**: [项目页](https://follow-your-shape.github.io/)  
**领域**: 扩散模型 / 图像编辑  
**关键词**: Shape Editing, Trajectory Divergence Map, Training-Free, Flow Matching, KV Injection  

## 一句话总结
提出 Follow-Your-Shape，一个无需训练和掩码的形状感知编辑框架，通过计算反演与编辑轨迹间的 token 级速度差异构建 Trajectory Divergence Map (TDM) 来精确定位编辑区域，配合分阶段 KV 注入实现大幅形状变换且严格保持背景。

## 研究背景与动机
1. **领域现状**：基于扩散/Flow 模型的图像编辑在通用任务上表现良好，但在涉及大尺度形状变换的结构性编辑中常常失败——要么无法实现目标形状变化，要么破坏非编辑区域。
2. **现有痛点**：现有区域控制策略存在根本缺陷——
   - 外部二值掩码：过于刚性，难以处理精细边界
   - 交叉注意力图推断：噪声大且不稳定，对大形变不可靠
   - 无条件 KV 注入：全局保持结构但抑制目标编辑
3. **核心矛盾**：编辑可控性与内容保持之间的冲突——要让 Flow 模型精确修改目标区域形状，同时不影响其他区域。
4. **本文要解决什么？** 如何在无需训练、无需掩码的情况下实现精确的大尺度形状编辑？
5. **切入角度**：从动力系统视角——编辑区域可由源条件和目标条件下去噪轨迹的分歧程度来定位。
6. **核心idea一句话**：通过比较源和目标 prompt 下的 velocity field 差异自动定位编辑区域，用分阶段 KV 注入实现稳定的形状感知编辑。

## 方法详解
### 整体框架
基于 FLUX.1-dev 的推理时编辑框架。给定源图像和 prompt，先反演得到噪声 latent，然后分三阶段去噪：Stage 1 全局 KV 注入稳定轨迹 → Stage 2 编辑并收集 TDM → Stage 3 基于 TDM 掩码的选择性 KV 注入 + ControlNet 结构引导。

### 关键设计
1. **Trajectory Divergence Map (TDM)**：核心创新。在去噪过程中，计算源 prompt 和目标 prompt 条件下 velocity field 的 token 级 $L_2$ 差异：
$$\delta_t^{(i)} = \| v_\theta(\mathbf{z}_t^{(i)}, t, \mathbf{c}_{\text{tgt}}) - v_\theta(\mathbf{x}_t^{(i)}, t, \mathbf{c}_{\text{src}}) \|_2$$
经 min-max 归一化后得到 $\tilde{\delta}_t^{(i)} \in [0,1]$，分歧大的区域即为需要编辑的区域。

2. **分阶段编辑策略（Staged Editing）**：
   - **Stage 1（初始轨迹稳定）**：前 $k_{\text{front}}=2$ 步进行无条件 KV 注入（$M_S = \mathbf{0}$），将轨迹锚定到忠实重建流形上
   - **Stage 2（编辑 + TDM 聚合）**：在预定窗口 $N$ 中设置 $M_S = 1$ 允许编辑，同时收集各时间步的归一化 TDM。使用 softmax 加权时序融合聚合：
   $$\hat{\delta}^{(i)} = \sum_{t \in N} \alpha_t^{(i)} \cdot \tilde{\delta}_t^{(i)}, \quad \alpha_t^{(i)} = \frac{\exp(\tilde{\delta}_t^{(i)})}{\sum_{t'} \exp(\tilde{\delta}_{t'}^{(i)})}$$
   再经高斯模糊 $\tilde{M}_S = \mathcal{G}_\sigma * \hat{\delta}$ 并用 Otsu 阈值法得到二值掩码 $M_S$
   - **Stage 3（结构与语义一致性）**：根据 $M_S$ 混合 KV 特征（编辑区域用目标 KV，非编辑区域用源 KV）：
   $$\{K^*, V^*\} \leftarrow M_S \odot \{K^{\text{tgt}}, V^{\text{tgt}}\} + (1 - M_S) \odot \{K^{\text{inv}}, V^{\text{inv}}\}$$
   同时可选 ControlNet（depth + Canny）提供辅助结构约束

3. **ReShapeBench 基准**：120 张新图像，分为单目标（70）和多目标（50）场景 + 通用集（50），共 290 个形状编辑用例。源-目标 prompt 仅在前景对象描述上不同，经人工验证。

### 损失函数 / 训练策略
- **无训练方法**：纯推理时框架，不涉及训练或微调
- 14 步去噪，guidance scale 2.0，$k_{\text{front}} = 2$
- ControlNet 条件应用于归一化去噪区间 $[0.1, 0.3]$，depth 强度 2.5，Canny 强度 3.5
- 使用 RF-Solver 二阶反演方案

## 实验关键数据
### 主实验（ReShapeBench + PIE-Bench）

| 方法 | AS↑ | PSNR↑ | LPIPS↓(×10³) | CLIP Sim↑ |
|------|-----|-------|-------------|-----------|
| MasaCtrl | 5.83 | 23.54 | 125.36 | 20.84 |
| RF-Edit | 6.52 | 33.28 | 17.53 | 30.41 |
| KV-Edit | 6.51 | 34.73 | 16.42 | 26.97 |
| FLUX.1 Kontext | 6.53 | 32.91 | 18.35 | 28.53 |
| Ours (w/o ControlNet) | 6.52 | 34.85 | **9.04** | 32.97 |
| **Ours (Full)** | **6.57** | **35.79** | **8.23** | **33.71** |

*ReShapeBench 上 Follow-Your-Shape 全面领先，LPIPS 仅 8.23（次优 16.42），CLIP Sim 33.71（次优 30.41）。*

### 消融实验（$k_{\text{front}}$ 的影响）

| $k_{\text{front}}$ | AS↑ | PSNR↑ | LPIPS↓(×10³) | CLIP Sim↑ |
|------|-----|-------|-------------|-----------|
| 0 | 6.51 | 32.79 | 10.04 | 31.05 |
| 1 | 6.55 | 34.38 | 9.88 | 32.56 |
| **2** | **6.57** | **35.79** | **8.23** | **33.71** |
| 3 | 6.52 | 31.25 | 10.52 | 29.41 |
| 4 | 6.48 | 30.41 | 12.37 | 27.66 |

*$k_{\text{front}}$ 存在最优点：太小则轨迹不稳定，太大则抑制编辑——$k_{\text{front}}=2$ 最佳平衡。*

### 关键发现
- 即使不用 ControlNet，TDM 引导的 KV 注入已显著优于所有 baseline
- 扩散模型方法（MasaCtrl、PnPInversion、Dit4Edit）在大形变下严重退化
- Flow 模型方法虽图像质量更高，但在剧烈形状变换中仍有鬼影/不完全变换
- Otsu 阈值法自适应且无需人工调参，TDM 分布天然适合二分割
- ControlNet 时机（$[0.1, 0.3]$）和强度（depth 2.5, Canny 3.5）的合理选择很关键

## 亮点与洞察
- **动力系统视角**：将编辑区域定位问题转化为轨迹分歧度量，理论直觉优雅
- **完全避免外部掩码和注意力图**：TDM 从模型自身行为中动态提取编辑区域
- **三阶段调度设计精妙**：稳定→探索→约束的流程符合去噪动态特性
- **softmax 加权时序融合**：比简单平均更好地捕捉时间动态（某 token 可能在某些时间步不变但在后续时间步变化）
- **ReShapeBench 填补了形状编辑评估的空白**：现有 benchmark 未针对大尺度形状变换设计

## 局限性 / 可改进方向
- 仅支持基于 prompt 的形状编辑，无法处理无法用文字精确描述的复杂几何变换
- 需要求解二阶反演（RF-Solver），计算量翻倍
- TDM 在高噪声阶段不可靠，需要分阶段策略规避——但这增加了超参数（$k_{\text{front}}$, 窗口大小 $N$）
- 对 ControlNet 的依赖是可选的但在某些场景下仍需要，偏离了"完全无外部工具"的理想
- 未探索在视频编辑中的扩展

## 相关工作与启发
- **RF-Edit / RF-Solver**：本文构建在 RF-Solver 反演之上，用 TDM 替代其全局 KV 注入策略
- **DiffEdit**：同样计算源-目标预测差异生成掩码，但 DiffEdit 基于扩散模型且不考虑轨迹时序动态
- **Stable Flow**：提出 vital layers 的选择性注入，Follow-Your-Shape 在空间维度做选择性注入
- 启发：Flow Matching 的确定性 ODE 轨迹天然适合计算分歧度量，这种"轨迹-based"范式可能推广到视频编辑或 3D 内容编辑

## 评分
- 新颖性: ⭐⭐⭐⭐ — TDM 的轨迹分歧视角新颖，但分阶段注入是渐进式创新
- 实验充分度: ⭐⭐⭐⭐ — 双 benchmark + 详细消融，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐⭐ — 动机图示清晰，方法推导流畅
- 价值: ⭐⭐⭐⭐ — 为形状编辑这一被忽视的子任务提供了系统解决方案
