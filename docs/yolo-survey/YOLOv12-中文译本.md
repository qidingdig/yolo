# YOLOv12：以注意力为中心的实时目标检测器（中文译本）

> 原文：*YOLOv12: Attention-Centric Real-Time Object Detectors*  
> 作者：Yunjie Tian, Qixiang Ye, David Doermann  
> 来源：[arXiv:2502.12524](https://arxiv.org/abs/2502.12524)（2025，技术报告）  
> 代码：https://github.com/sunsmarterjie/yolov12  
> 说明：论文式中文整理译本；公式以图片嵌入。

**生产提示（Ultralytics）：** 官方对比页曾提示 YOLO12 注意力层可能导致训练不稳、显存偏高、CPU 推理偏慢，**生产环境需谨慎评估**。本译本保留学术要点供精读。

---

## 摘要

YOLO 长期以 CNN 改架构为主，尽管注意力建模更强，却因二次复杂度与低效内存访问难以匹敌 CNN 速度。本文提出 **注意力中心** 的 **YOLOv12**：在保持 YOLO 级速度的同时吸收注意力收益。

例：YOLOv12-N 达 **40.6% mAP**，T4 延迟约 **1.64 ms**，相对 YOLOv10-N / YOLO11-N 约 **+2.1 / +1.2 mAP**，速度可比。相对 RT-DETR-R18，YOLOv12-S 更快约 **42%**，算力/参数约仅其 **36% / 45%**。

---

## 1. 引言

注意力慢的主因：

1. **复杂度** \(\mathcal{O}(L^2 d)\) vs 卷积线性；  
2. **内存访问**：注意力中间图在 HBM 与 SRAM 间往返（FlashAttention 可缓解）。

YOLOv12 三招：

1. **Area Attention（A2）**：简单分区降注意力成本，保大感受野；  
2. **R-ELAN**：块级残差 + 缩放，改善注意力（尤其大模型）优化；  
3. **架构适配**：FlashAttention、去掉位置编码、MLP ratio 调至约 1.2、减少堆叠深度、尽量用卷积提速。

家族：N / S / M / L / X，评测设定对齐 YOLO11，无额外预训练花招。

---

## 2. 方法

### 2.1 Area Attention

<p align="center">
  <img src="figures/yolov12/eq_area.png" alt="area attention" width="680" />
</p>

不做复杂窗口划分：把特征图沿水平或垂直**等分为 \(l\) 个区域**（默认 4），在区域内做注意力。相对全局注意力大幅降成本，相对繁琐局部注意力实现更干净、感受野仍大。

### 2.2 R-ELAN

<p align="center">
  <img src="figures/yolov12/eq_arch.png" alt="YOLOv12 / R-ELAN" width="720" />
</p>

相对 ELAN / C3k2：

1. **块级残差 + scaling**：稳定注意力带来的优化难度；  
2. **重设计特征聚合**：更适配注意力中心堆叠。

### 2.3 其他工程化改动

- 引入 **FlashAttention** 缓解 I/O 瓶颈；  
- 去掉位置编码等拖慢模块；  
- MLP 比例从常见 4 调到约 **1.2**，平衡注意力与 FFN 算力；  
- 减少堆叠深度以利优化；  
- 能用卷积处尽量用卷积。

---

## 3. 实验（摘要）

640 输入，论文 Table 1（T4 延迟量级）：

| 模型 | FLOPs | 参数 | AP<sup>val</sup> | Latency (ms) |
|------|-------|------|------------------|--------------|
| YOLOv12-N | 6.5G | 2.6M | **40.6%** | 1.64 |
| YOLOv12-S | 21.4G | 9.3M | **48.0%** | 2.61 |
| YOLOv12-M | 67.5G | 20.2M | **52.5%** | 4.86 |
| YOLOv12-L | 88.9G | 26.4M | **53.7%** | 6.77 |
| YOLOv12-X | 199.0G | 59.1M | **55.2%** | 11.79 |

对照：YOLO11-N 39.4% @ 1.5 ms；YOLOv10-N 38.5% @ 1.84 ms。

---

## 4. 结论

YOLOv12 证明：通过 **Area Attention + R-ELAN + FlashAttention 等适配**，注意力可以进入 YOLO 实时区间，并在延迟—精度与 FLOPs—精度上刷新当时多项对比。落地时需单独评估训练稳定性与 CPU 场景。

---

## 译本说明

数字以 arXiv:2502.12524 为准；与 Ultralytics 集成版超参/导出行为可能略有差异。
