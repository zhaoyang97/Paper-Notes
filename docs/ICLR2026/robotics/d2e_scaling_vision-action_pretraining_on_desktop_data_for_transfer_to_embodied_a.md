# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI

**会议**: ICLR 2026
**arXiv**: [2510.05684](https://arxiv.org/abs/2510.05684)
**代码**: [项目页](https://worv-ai.github.io/d2e/)
**领域**: 机器人
**关键词**: embodied AI, desktop pretraining, inverse dynamics model, vision-action pretraining, robotics transfer

## 一句话总结
提出 D2E 框架，证明桌面游戏交互数据可作为具身 AI 的有效预训练基底：通过 OWA 工具包收集 335h 人类演示 + Generalist-IDM 伪标注 1000+h YouTube 游戏视频 + VAPT 迁移训练，1B 参数模型在 LIBERO 操作达 96.6%、CANVAS 导航达 83.3%，匹敌或超越 7x 更大的模型。

## 研究背景与动机
1. **领域现状**: LLM 得益于互联网规模文本数据实现了跨任务泛化，但具身 AI 的物理轨迹数据收集成本极高（专用硬件、人工操作、复杂标注），数据规模远不足以驱动类似的 scaling。
2. **现有痛点**: 现有机器人数据集（DROID 等）规模小、领域特定、格式不兼容。VPT 仅限 Minecraft 单域，SIMA 跨游戏但数据私有。
3. **核心矛盾**: 具身 AI 需要大规模动作标注数据，但物理数据收集不可扩展；桌面交互（键盘鼠标）丰富且标准化，但能否迁移到物理机器人？
4. **本文要解决什么**: 建立从桌面数据收集到具身任务迁移验证的完整管线。
5. **切入角度**: 游戏交互具有复杂感觉运动模式（导航、操作、规划），与具身 AI 挑战高度类似，且可通过 YouTube 大规模获取。
6. **核心idea一句话**: 桌面 = 便宜的具身 AI 预训练数据源，OWA收集 + 通用IDM伪标注 + VAPT迁移 = 完整管线。

## 方法详解
### 整体框架
三组件管线：(1) OWA Toolkit 数据收集与格式化 → (2) Generalist-IDM 伪标注 YouTube 视频 → (3) VAPT 在桌面数据上预训练后迁移到机器人任务。

### 关键设计
1. **OWA Toolkit**:
   - **ocap 录制器**: 基于 Windows API + GStreamer 同步录制屏幕(60Hz)、键盘、鼠标事件，精确时间对齐
   - **OWAMcap 格式**: 扩展 MCAP 标准，支持 H.265 编码（217x 压缩 vs raw），MediaRef 外部引用实现高效随机访问
   - **数据管线优化**: FSLDataset（固定序列长度打包）+ 自适应批量解码，吞吐量 119 img/s（10.2x 提升），平均磁盘读取仅 18.73 KB/img（比 TorchCodec 低 41x）
   - 14名标注员1个月收集 335h 数据跨 31 款游戏（DROID 需50名收集者×13机构×12个月）

2. **Generalist-IDM**:
   - **时间戳事件标记化**: 每个事件序列化为 `<EVENT_START>{TYPE}{TIMESTAMP}{DETAIL}<EVENT_END>`，不依赖固定 tick 间隔
   - **NEP-τ (带时间偏移的下一事件预测)**: 将观测窗口向前移动 $\tau$ 步，提供未来上下文：
     $$\mathcal{L}_{\text{NEP-}\tau} = -\mathbb{E}\left[\sum_t \log P_\theta(a_t | o_{1:\min(t+\tau,T)}, a_{1:t-1})\right]$$
     $\tau=100$ms 为最优，$\tau=0$ 时 Pearson 相关性几乎为零
   - 基于 InternVL3-1B 架构，仅需 ~192 H100小时（~$800）训练
   - 零样本泛化到未见游戏，键盘准确率在 Battlefield 6 达 63%（匹配专用 IDM）

3. **VAPT (Vision-Action PreTraining)**:
   - 在 1.3K+ 小时桌面数据（259h 人工 + 1000+h 伪标注）上预训练 InternVL3-1B
   - 迁移机制假说：(i) 动作模态对齐，(ii) 目标导向序列决策，(iii) 高多样性跨20游戏
   - 训练损失曲线显示 VAPT 初始化模型立即收敛，而基线有初始平台期

### 损失函数 / 训练策略
- Generalist-IDM: NEP-τ 目标（自回归 next-event prediction）
- VAPT: 标准视觉-动作预训练目标，下游任务微调

## 实验关键数据
### 主实验
LIBERO 操作基准（成功率%）:

| 方法 | 参数量 | Spatial | Object | Goal | 10(long) | Total |
|------|--------|---------|--------|------|----------|-------|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 3.3B | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 |
| SmolVLA | 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.7 |
| **VAPT w/o pseudo** | **1B** | 95.8 | 98.4 | **98.6** | **93.6** | **96.6** |

CANVAS 导航基准（Baseline 75.3% → VAPT w/ pseudo **83.3%**）。

### 消融实验
Generalist-IDM vs Specialist-IDM（域内6款游戏）:

| 游戏 | 模型 | Pearson-X | 键盘准确率 |
|------|------|-----------|----------|
| Brotato | Specialist | 65.92 | 28.80 |
| Brotato | **Generalist** | **73.65** | **86.36** |
| Apex Legends | Specialist | 65.16 | 67.47 |
| Apex Legends | **Generalist** | **83.90** | **76.55** |

Generalist-IDM 在所有游戏上均超越 Specialist-IDM（键盘准确率最高提升 57.6%）。

### 关键发现
- 1B 参数 VAPT 在 LIBERO 上超越 3.3B π₀ 和 7B OpenVLA，长时域任务优势尤其明显（93.6% vs 73.0%）
- 伪标注数据对**导航**有帮助（+8%），但对**操作**反而无益——操作需要精确人工监督
- 真实机器人（SO101 pick-and-place）从 70% 提升到 80%，验证物理迁移有效
- OWA 数据收集效率远超机器人数据：14人×1月 vs DROID的50人×13机构×12月

## 亮点与洞察
- **范式创新**: 首次系统性地证明桌面游戏→物理机器人的迁移是可行的
- **完整管线**: 从工具→数据→模型→迁移→验证的端到端贡献
- **OWA Toolkit 的工程价值**: 152x 压缩、41x I/O 优化、标准化格式，对社区有实际贡献
- **NEP-τ 设计**: 时间戳驱动的事件预测比 tick-based 更高效，跳过无操作时段
- **Generalist-IDM 的零样本泛化**: 在未见游戏上超越专用模型，使互联网规模伪标注成为可能

## 局限性 / 可改进方向
- 桌面→机器人的迁移机制仍是假说层面，缺乏严格的理论分析
- 仅验证了 InternVL3-1B 一种骨干
- 真实机器人实验仅做了单个 pick-and-place 任务
- OWA Toolkit 目前仅支持 Windows
- 游戏数据可能引入与真实世界不一致的视觉偏见
- Meta-World 上绝对成功率仍较低（Very Hard 仅 20-24%）

## 相关工作与启发
- VPT 是先驱但限于 Minecraft 单域；SIMA 跨游戏但私有；D2E 是首个开源、多游戏、验证迁移的框架
- 与 RT-X/Open X-Embodiment 互补：D2E 提供了低成本预训练数据源
- LAPA 的潜动作预训练思路类似，但 D2E 直接操作在显式键鼠动作上
- 启发：互联网视频 + 通用 IDM 伪标注是扩展具身 AI 数据的可行路径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 桌面→具身的迁移范式具有开创性意义
- 实验充分度: ⭐⭐⭐⭐ LIBERO/Meta-World/CANVAS + 真实机器人，但真实机器人实验较简单
- 写作质量: ⭐⭐⭐⭐ 系统性强，但内容跨度大导致部分细节需查附录
- 价值: ⭐⭐⭐⭐⭐ 开源工具+数据+模型，为具身AI数据扩展开辟新方向
