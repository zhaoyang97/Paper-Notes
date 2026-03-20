# RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

**会议**: CVPR 2026  
**arXiv**: [2603.11106](https://arxiv.org/abs/2603.11106)  
**代码**: [https://heikaishuizz.github.io/RC-NF/](https://heikaishuizz.github.io/RC-NF/)  
**领域**: 具身智能 / 机器人操纵 / 异常检测  
**关键词**: 归一化流, 机器人异常检测, VLA模型, OOD检测, 实时监控  

## 一句话总结

提出RC-NF，一种基于条件归一化流的实时异常检测模型，通过解耦处理机器人状态和物体轨迹特征，仅需正样本无监督训练即可在100ms内检测VLA模型执行中的OOD异常，在LIBERO-Anomaly-10上以约8% AUC和10% AP的优势超越SOTA（包括GPT-5、Gemini 2.5 Pro等VLM基线）。

## 背景与动机

VLA（Vision-Language-Action）模型通过模仿学习在专家演示数据上训练，能执行复杂操纵任务，但在动态真实环境中部署时经常遇到OOD场景导致失败。现有的运行时监控方案有两类痛点：（1）**状态分类方法**（行为树等）需要穷举所有异常条件或手动定义前置条件，难以应对现实中的组合变异；（2）**VLM方法**（如Sentinel用GPT-5/Gemini做推理）需要多步推理，延迟达数秒级别，无法及时干预。因此需要一个既准确又快速（<100ms）的轻量级监控器。

## 核心问题

如何为VLA模型设计一个**即插即用**的实时监控模块，能在不修改VLA架构的前提下：（1）仅用正常演示数据训练（无需收集失败数据）；（2）实时判断机器人状态-物体轨迹的联合分布是否偏离正常任务分布；（3）支持任务级（task-level）和状态级（state-level）两种粒度的OOD检测与修正？

## 方法详解

### 整体框架

RC-NF的pipeline如下：视频流经SAM2提取物体分割mask，通过网格采样得到点集表示（point set）；同时获取机器人关节/夹爪/位姿状态和任务文本描述。这些输入送入条件归一化流模型，经K步可逆变换映射到高斯潜空间，通过负对数似然计算异常分数。当分数超过阈值时触发任务级重规划或状态级回滚。

### 关键设计

1. **球面均匀编码的任务嵌入（Spherical Uniform Encoding）**: 将不同任务指令映射到T维超球面上均匀分布的向量，确保任务嵌入在潜空间中最大化分离。这种设计使模型不仅能检测数据集级OOD，还能区分任务特定的异常（如Spatial Misalignment中机器人走错方向）。消融显示去掉任务嵌入后Spatial Misalignment的AUC从0.97骤降至0.81。

2. **RCPQNet（Robot-Conditioned Point Query Network）**: 这是RC-NF的核心仿射耦合层设计。机器人状态经线性投影后通过FiLM机制与任务嵌入调制，生成Task-aware Robot-Conditioned Query token；物体点集则通过双分支编码（Dynamic Shape分支做中心化归一化提取形状特征，Positional Residual分支补偿位置信息），经MLP+GRU+Transformer编码为memory token。最终通过交叉注意力融合，生成仿射变换的scale和shift参数。这种解耦设计避免了FailDetect中直接拼接导致的特征干扰。

3. **双分支点特征编码（Dual-Branch Point Feature Encoding）**: Dynamic Shape分支对每帧点集做中心化和归一化，消除平移/缩放影响，捕捉物体的形状变化；Positional Residual分支保留原始坐标信息，补偿归一化损失。两者分别经GRU建模时序依赖后由Transformer编码输出。消融显示Dynamic Shape分支是最关键组件（去掉后AUC从0.93降至0.68），而Positional Residual分支提供互补（去掉后AUC降至0.89）。

4. **异常检测与处理（Anomaly Detection & Handling）**: 推理时用负对数似然作为异常分数，阈值通过conformal prediction在校准集上估计（均值+分位数）。检测到异常后区分两种情况：**任务级OOD**（环境变化导致指令失效，如抽屉关闭）→ 暂停并触发高层重规划；**状态级OOD**（任务仍有效但轨迹偏移，如物体滑落）→ 激活homing程序回归初始状态后继续执行。

### 损失函数 / 训练策略

- **训练目标**: 最大化条件对数似然 $\log p_{X|C}(x|c)$，其中高斯先验的均值使用任务嵌入 $\mu_{\text{task}}$
- **BalancedHardSampler去偏**: 机器人初始运动阶段轨迹相似度高导致样本不均衡，设计分阶段训练策略——先标准采样训练至NextStageEpoch，再用BalancedHardSampler平衡样本分布减少冗余
- 训练配置：K=12流步，100 epochs

## 实验关键数据

| 数据集 | 异常类型 | 指标 | RC-NF | GPT-5 | FailDetect | 提升(vs best) |
|--------|----------|------|-------|-------|------------|--------------|
| LIBERO-Anomaly-10 | Gripper Open | AUC | 0.9312 | 0.9137 | 0.7883 | +1.9% |
| LIBERO-Anomaly-10 | Gripper Slippage | AUC | 0.9195 | 0.8941 | 0.6665 | +2.8% |
| LIBERO-Anomaly-10 | Spatial Misalign. | AUC | 0.9676 | 0.5292 | 0.6557 | +31.2% |
| LIBERO-Anomaly-10 | Average | AUC/AP | 0.9309/0.9494 | 0.8500/0.8507 | 0.7181/0.7700 | +8%/+10% |

**实时性能**: RC-NF在RTX 3090上单帧推理总延迟86.7ms（SAM2 50ms + 网格采样1.7ms + RC-NF 30ms + 其他5ms），满足实时要求。

### 消融实验要点

- **任务嵌入是区分任务特定异常的关键**：去掉后Spatial Misalignment从0.97降至0.81，只能检测数据集级OOD
- **机器人状态对Gripper Open类异常至关重要**：去掉后该类AUC从0.93骤降至0.63，因为夹爪未闭合不改变物体位置，异常仅体现在机器人-物体相对运动中
- **Dynamic Shape分支贡献最大**（去掉AUC降至0.68），Positional Residual分支互补（去掉AUC降至0.89）
- VLM基线在Spatial Misalignment上几乎失效（AUC≈0.5），说明空间推理是VLM的弱点

## 亮点

- **概率框架的优雅性**: 用归一化流做异常检测是natural fit——正常行为对应高概率密度，异常对应低密度，无需负样本
- **解耦设计的精巧**: RCPQNet将机器人状态作为query、物体特征作为memory，通过交叉注意力实现解耦交互，比FailDetect的简单拼接更有效
- **双粒度异常处理**: 任务级和状态级OOD的区分与对应处理策略（重规划 vs 回滚）在工程上非常实用
- **即插即用**: 不修改VLA架构，与π0等模型并行运行，设计理念优于依赖VLM的方案

## 局限性 / 可改进方向

- Benchmark仅在LIBERO-10上构建，任务和场景多样性有限，需要在更多真实任务上验证
- 阈值设定依赖校准数据集，不同任务可能需要不同的α参数
- SAM2的物体分割质量会影响点集表示，遮挡或快速运动场景可能受限
- 目前假设异常后可以通过homing或重规划恢复，对不可逆异常（如物体损坏）未讨论
- 球面均匀编码的任务数量可扩展性未验证（当前仅10个任务）

## 与相关工作的对比

- **vs FailDetect**: 同为flow-based方法，但FailDetect直接拼接图像特征和机器人状态作为flow matching输入，导致特征纠缠和不平衡。RC-NF通过RCPQNet解耦处理+交叉注意力融合，在特征选择（点集代替原始图像）和处理（解耦代替拼接）上均有本质改进
- **vs VLM监控（Sentinel + GPT-5/Gemini）**: VLM方案在空间推理上几乎失效（Spatial Misalignment AUC≈0.5），且多步推理延迟达秒级。RC-NF通过概率密度替代语义理解，在空间异常上表现突出（AUC=0.97），且延迟<100ms
- **vs 行为树方法**: 行为树需要显式设计异常条件和回滚步骤，RC-NF仅需正常演示数据即可无监督学习

## 启发与关联

- 归一化流的条件概率密度估计思路可推广到其他需要实时异常检测的场景（自动驾驶、医疗手术机器人等）
- 解耦query-memory的交叉注意力设计在多模态特征融合中有普遍参考价值
- 点集表示作为物体状态的中间表征，比raw image更鲁棒，这个思路可用于其他embodied AI任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 条件归一化流用于机器人监控不算全新，但RCPQNet解耦设计和双分支编码有实质贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 提出新benchmark，对比VLM和flow-based基线，消融完整，有真机验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图示精美，motivation-method-experiment逻辑连贯
- 价值: ⭐⭐⭐⭐ 解决VLA部署中的关键安全问题，即插即用设计有工程价值
