# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

**会议**: ICCV 2025  
**arXiv**: [2503.19757](https://arxiv.org/abs/2503.19757)  
**代码**: [https://robodita.github.io/](https://robodita.github.io/)  
**领域**: 多模态VLM / 具身智能 / 机器人策略  
**关键词**: VLA, diffusion policy, transformer, cross-embodiment, robot manipulation, in-context conditioning  

## 一句话总结
提出Dita，用Transformer架构进行统一的多模态扩散过程直接去噪连续动作序列，通过in-context conditioning实现去噪动作与历史视觉观察的细粒度对齐，在跨embodiment数据集上scaling后实现SOTA仿真性能和10-shot真实世界长horizon任务适应。

## 背景与动机
当前VLA（Vision-Language-Action）模型如RT-2、OpenVLA等在跨机器人泛化上取得进展，但它们依赖紧凑的动作头（action head）来预测离散化或连续动作，限制了对异构动作空间的适应性。不同机器人的动作空间差异很大（关节数、自由度、coordinate frame等），简单的MLP动作头难以统一处理。Diffusion Policy证明了扩散模型在动作生成上的强大能力，但如何将其与Transformer的scaling能力结合、并在跨embodiment数据上有效训练是核心挑战。

## 核心问题
如何设计一个可扩展的VLA框架，通过统一的扩散过程处理来自不同机器人的异构动作空间，并在scaling up后展现强泛化能力？

## 方法详解

### 整体框架
Dita将VLA问题分解为：视觉编码器提取观察特征 → LLM处理语言指令和历史上下文 → Diffusion Transformer去噪生成动作序列。关键区别在于动作去噪不是用简单的MLP，而是用完整的Transformer实现。

### 关键设计
1. **In-Context Conditioning**：不像之前方法将所有条件信息（视觉+语言）先融合为一个固定embedding再送入去噪网络，Dita让去噪Transformer直接与原始视觉tokens做cross-attention——被去噪的动作token可以在每个denoising step自由attend到任何历史观察的任何视觉patch。这实现了动作与环境的细粒度对齐，使模型能精确感知action delta和环境细微变化。

2. **Transformer动作去噪器的Scaling**：利用Transformer天然的scalability，动作去噪器可以随参数量增加而提升——与Transformer在语言/视觉中的scaling法则一致。这使得Dita能有效整合来自不同embodiment的跨数据集训练。

3. **跨Embodiment统一**：通过扩散过程生成连续动作序列（而非离散token），Dita可以自然适应不同机器人的不同维度动作空间。不同embodiment的动作通过padding/masking统一到同一维度，扩散去噪过程在统一空间中进行。

### 损失函数 / 训练策略
标准的扩散去噪loss（velocity prediction或epsilon prediction），在跨embodiment数据集上联合训练，支持不同相机视角、观察场景和动作空间。

## 实验关键数据
- 在多个仿真benchmark上达到SOTA或comparable性能
- 真实世界：通过仅**10-shot微调**，仅使用第三人称相机输入，成功适应环境变化和复杂长horizon任务
- 展现了对不同相机视角、场景变化和任务复杂度的鲁棒性
- 轻量级且开源的通用机器人策略baseline

### 消融实验要点
- In-context conditioning >> Fused embedding conditioning（动态attend优于固定embedding）
- Scaling up denoiser Transformer持续提升性能
- 跨embodiment联合训练提升每个单一embodiment的性能（正迁移）
- 10-shot微调在真实世界足够有效

## 亮点
- **In-context conditioning是关键创新**：让动作去噪直接attend到原始视觉tokens，比先融合再去噪更精细——这对需要精确操控的任务尤其重要
- **Transformer去噪器的scalability**：验证了DiT在机器人动作预测领域的scaling潜力
- **10-shot真实世界适应**：极低的数据需求使方案实用性极强
- **开源baseline**：提供了一个轻量、通用的VLA框架供社区使用

## 局限性 / 可改进方向
- 扩散推理的多步去噪在实时控制中可能有延迟问题
- 仅用10-shot微调的泛化范围可能有限（未测试全新任务类型）
- 第三人称相机限制——自对心等精密操作可能需要手部相机
- 未与最新的大规模VLA（如pi0等）系统对比

## 与相关工作的对比
- **vs. OpenVLA/RT-2**：这些用离散化或MLP动作头；Dita用Transformer扩散去噪——更适合复杂连续动作空间
- **vs. Diffusion Policy**：Diffusion Policy用UNet做去噪；Dita用Transformer做去噪并加入in-context visual conditioning——更scalable
- **vs. GTR**：GTR解决VLM agent的思维坍塌（RL层面）；Dita提供了更好的VLA架构（模型层面）——互补

## 启发与关联
- In-context conditioning的思路可以迁移到其他条件生成任务——如text-guided视频生成中让去噪网络直接attend原始文本tokens
- 与SANA-Sprint结合：如果能将动作扩散去噪也加速到1-4步，机器人控制的实时性将大幅提升

## 评分
- 新颖性: ⭐⭐⭐⭐ In-context conditioning和Transformer动作去噪的结合有新意
- 实验充分度: ⭐⭐⭐⭐ 仿真benchmark全面，真实世界10-shot验证有说服力
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰
- 价值: ⭐⭐⭐⭐ 开源通用VLA baseline，对机器人学习社区有直接贡献
