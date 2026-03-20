# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**代码**: [https://dreamvideo-omni.github.io](https://dreamvideo-omni.github.io/)  
**领域**: 视频生成 / 可控生成  
**关键词**: video customization, multi-subject, motion control, identity preservation, reward learning  

## 一句话总结
统一框架同时实现多主体身份定制和全运动控制（全局运动 + 局部运动 + 相机运动），通过渐进式两阶段训练（有监督微调 + 潜空间身份奖励反馈学习）解决身份保持与运动控制之间的固有冲突。

## 背景与动机
视频生成中同时做到"多主体身份保持"和"多粒度运动控制"是未解决的挑战。现有方法分成两个独立方向：主体定制（保持外观但运动不可控）和运动控制（精确运动但不能指定主体外观），少数尝试统一的方法在三方面碰壁：（1）运动控制粒度有限——大多只支持单一信号（bbox / depth / 稀疏轨迹）；（2）多主体控制歧义——各条件无显式绑定，模型不知道哪个运动属于哪个主体；（3）身份退化——运动控制要求像素动态变化，而身份保持要求与静态参考一致，标准扩散重建损失无法调和这一矛盾。

## 核心问题
如何在单一 DiT 框架中同时实现稳健的多主体身份保持和覆盖全局运动、局部动态、相机运动的全方位控制，并解决两者之间的固有冲突？

## 方法详解

### 整体框架
基于 Wan2.1-1.3B T2V DiT，渐进式两阶段训练：
- **Stage 1（Omni-Motion & Identity SFT）**：集成 ⟨参考图像, 全局 Bbox, 局部轨迹⟩ 三元组进行联合训练，包含单/多主体定制、全局/局部运动控制、相机运动控制等多任务
- **Stage 2（Latent Identity Reward Feedback Learning）**：训练潜空间身份奖励模型 LIRM，在潜空间直接提供身份保持的奖励信号，绕过昂贵的 VAE 解码

### 关键设计

1. **条件感知 3D RoPE**：不同类型输入（视频帧、参考图像、padding、轨迹）分配不同的时间索引策略。视频帧用连续序列索引，参考图像用共享固定索引（标记为静态条件），padding 用无效索引，轨迹与视频帧共享索引保持时空对齐。去掉此模块会导致训练崩溃。

2. **Group & Role Embeddings**：Group embedding 为每个 ⟨参考, bbox, 轨迹⟩ 三元组分配唯一标识，确保运动信号与正确主体绑定；Role embedding 区分"外观资产"（参考图像→object embedding）和"控制信号"（bbox/轨迹→control embedding）。消融显示去掉后运动控制精度大幅下降，尤其多主体模式。

3. **层级运动注入（Hierarchical BBox Injection）**：bbox 潜变量通过 layer-wise zero-convolutions 注入到每个 DiT block 的输出，而非仅在输入层加一次。去掉后多主体 mIoU 从 0.570 暴跌到 0.289。

4. **潜空间身份奖励模型（LIRM）**：以预训练 VDM 前 8 层为 backbone + 身份交叉注意力层 + MLP 预测头。参考图像特征作为 Query，视频时空特征作为 Key/Value，通过交叉注意力计算身份对齐度。用 ~27,500 视频的人类偏好数据训练，BCE 损失优化。关键发现：参考图像必须作为 Query（作为 KV 则准确率从 0.720 暴跌到 0.455）。

5. **潜空间身份奖励反馈学习（LIReFL）**：在去噪过程中随机采样中间时间步 t_m，执行单步梯度可传播的去噪，将中间潜变量送入冻结的 LIRM 计算奖励。损失 = L_sft + 0.1 · L_LIReFL。完全绕过 VAE 解码器，在任意时间步提供密集反馈。

### 损失函数 / 训练策略
- Stage 1：重加权扩散损失 L_sft，bbox 内区域权重放大 λ₁=2。条件随机丢弃 p=0.5
- Stage 2 LIRM：BCE 损失，差异学习率（预测头 1e-5，VDM backbone 1e-6），冻结文本和 patch embedding，~4K 步
- Stage 2 LIReFL：L = L_sft + 0.1 · L_LIReFL，3400 步，lr=5e-6。λ₂=0.10 为最优，过大（1.0）会 reward hacking

## 实验关键数据

**DreamOmni Bench（1027 视频，436 单主体 + 591 多主体）：**

| 指标 | DreamVideo-2 | DreamVideo-Omni |
|------|-------------|-----------------|
| R-DINO | 0.429 | **0.499** |
| Face-S | 0.157 | **0.301** |
| mIoU | 0.212 | **0.558** |
| EPE | 24.05 | **9.31** |

**运动控制 vs 14B Wan-Move：**
- 单主体：mIoU 0.558 vs 0.507，EPE 9.31 vs 14.43（1.3B 参数超越 14B 模型）
- 多主体：mIoU 0.570 vs 0.541，EPE 6.08 vs 9.02

**用户研究**：在联合身份+运动对比中，Overall Quality 偏好率 89.2%（vs DreamVideo-2 的 10.8%）

**涌现能力**：虽然基于 T2V 模型，自动获得 zero-shot I2V 生成和首帧条件轨迹控制能力

### 消融实验要点
- 去掉 Cond-Aware 3D RoPE → 训练崩溃，R-CLIP 从 0.739 跌到 0.625
- 去掉 Group & Role Emb → 多主体 EPE 从 6.08 升到 20.69
- 去掉层级 BBox 注入 → 多主体 mIoU 从 0.570 跌到 0.289
- Stage 1 only (无 LIReFL) → Face-S 0.251 vs 全模型 0.301
- λ₂=1.0 → reward hacking，mIoU 跌到 0.350，EPE 飙升到 25.00
- 全时间步 ReFL vs 仅最后 3 步 → 全时间步在所有指标上更优

## 亮点
- 架构设计上每个组件都有明确的问题针对性，group/role embeddings 解决多主体歧义的方式直观高效
- LIRM 在潜空间运作，完全绕过 VAE 解码器，既节省计算又支持任意时间步反馈——这是对标准 ReFL 的重要改进
- 2M 视频级数据集构建流水线（RAFT→RAM++→Qwen3→GroundingDINO→SAM2→CoTracker3）工程量巨大但可复现
- 1.3B 模型在运动控制上超越 14B Wan-Move，参数效率极高

## 局限性 / 可改进方向
- 仅基于 1.3B 模型，扩展到更大规模 DiT 可能进一步提升质量
- 身份奖励模型训练依赖人工标注偏好数据（27.5K 视频对），扩展成本高
- 相机运动通过背景轨迹模拟，无显式 3D 相机参数建模，复杂相机运动可能不够精确
- 测试时需要用户同时提供参考图像、bbox 和轨迹，交互门槛较高

## 与相关工作的对比
- vs DreamVideo-2：后者仅支持单主体 + bbox 控制，DreamVideo-Omni 支持多主体 + 全运动控制，且所有指标全面领先
- vs Tora2：后者用解耦个性化提取器和门控自注意力，但未开源、不支持全运动控制
- vs Wan-Move(14B)：虽然 Wan-Move 视觉质量高，但轨迹遵循精度不足，且参数量是本方法的 ~11 倍
- vs Phantom/VACE：仅做主体定制不做运动控制，且本方法在多主体身份保持上更优

## 启发与关联
- 潜空间奖励建模的思路可推广到其他需要人类偏好对齐的生成任务（音频、3D 等）
- Group/Role embedding 的显式绑定机制可用于其他多条件控制的生成模型
- 利用 VDM 本身作为奖励模型 backbone（而非 CLIP/DINO）是值得关注的范式转变
- 训练范式的两阶段设计（SFT→RLHF）与 LLM 训练范式高度一致，体现了生成模型训练的收敛趋势

## 评分
- 新颖性: ⭐⭐⭐⭐ 潜空间身份奖励模型和 group/role embedding 是有意义的创新，整体框架整合度高
- 实验充分度: ⭐⭐⭐⭐⭐ 自建 benchmark、多维度对比、详尽消融、用户研究，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，每个设计选择都有消融验证，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 首个统一多主体定制与全运动控制的框架，实验结果令人信服，对视频生成领域有较大推动
