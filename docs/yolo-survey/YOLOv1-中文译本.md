# You Only Look Once：统一的实时目标检测

**原文：** You Only Look Once: Unified, Real-Time Object Detection  
**作者：** Joseph Redmon\*, Santosh Divvala\*\*\*, Ross Girshick\*\*\*\*, Ali Farhadi\*\*\*  
**单位：** University of Washington\*，Allen Institute for AI\*\*，Facebook AI Research\*\*\*\*  
**项目主页：** http://pjreddie.com/yolo/  
**原文链接：** https://arxiv.org/abs/1506.02640  

> 说明：本文为便于调研阅读的中文译本。
> **公式采用 Unicode / 纯文本写法**（放在代码块中），在 GitHub 的 Files changed、文件页、本地编辑器中都能直接阅读，不依赖公式渲染引擎。

---

## 摘要

我们提出 YOLO，一种新的目标检测思路。以往工作通常把分类器改造后用于检测；相反，我们将目标检测建模为对空间分离的边界框及其类别概率的回归问题。单个神经网络在一次前向计算中，直接从整幅图像预测边界框与类别概率。由于整个检测流水线就是一个网络，因此可以端到端地直接针对检测性能进行优化。

我们的统一架构速度极快。基础 YOLO 模型可以以每秒 45 帧实时处理图像。更小的 Fast YOLO 可达每秒 155 帧，同时仍能达到其他实时检测器两倍的 mAP。与最先进的检测系统相比，YOLO 的定位误差更多，但更不容易把背景误判为目标。最后，YOLO 学到的目标表征非常通用：当从自然图像泛化到艺术品等领域时，它优于包括 DPM 和 R-CNN 在内的其他检测方法。

---

## 1 引言

人类只需瞥一眼图像，就能立刻知道图像中有哪些物体、它们在哪里、以及它们如何相互作用。人类视觉系统快速且准确，使我们能够几乎不假思索地完成驾驶等复杂任务。快速、准确的目标检测算法将使计算机无需专用传感器即可驾驶汽车，使辅助设备能够向用户实时传达场景信息，并释放通用、可响应机器人系统的潜力。

当前的检测系统通常把分类器改造后用于检测。为了检测某个物体，这类系统会对测试图像中不同位置、不同尺度反复运行该物体的分类器。可变形部件模型（DPM）采用滑动窗口方式，在整幅图像上按均匀间隔运行分类器 [10]。

更近的方法如 R-CNN 使用区域提议：先在图像中生成潜在边界框，再对这些候选框运行分类器。分类之后，还需后处理来精修边界框、消除重复检测，并根据场景中其他物体重新打分 [13]。这些复杂流水线既慢，又难以优化，因为每个组件都必须单独训练。

我们把目标检测重塑为一个单一的回归问题：直接从图像像素到边界框坐标与类别概率。使用我们的系统，你只需“看一次”（You Only Look Once, YOLO）图像，就能预测其中有什么物体以及它们在哪里。

**图 1：YOLO 检测系统。** 用 YOLO 处理图像简单直接。我们的系统：(1) 将输入图像缩放到 448×448；(2) 在图像上运行单个卷积网络；(3) 按模型置信度对检测结果做阈值过滤。

YOLO 令人耳目一新地简单，见图 1。单个卷积网络同时预测多个边界框以及这些框的类别概率。YOLO 在整幅图像上训练，并直接优化检测性能。这一统一模型相较传统目标检测方法有若干优势。

第一，YOLO 极快。由于我们把检测建模为回归问题，因此不需要复杂流水线。测试时只需在新图像上运行神经网络即可得到检测结果。我们的基础网络在 Titan X GPU 上、无批处理时可达 45 FPS；快速版本超过 150 FPS。这意味着我们可以以低于 25 毫秒的延迟实时处理流视频。此外，YOLO 的平均精度均值（mAP）超过其他实时系统两倍以上。实时摄像头演示见项目主页：http://pjreddie.com/yolo/ 。

第二，YOLO 在做预测时对图像进行全局推理。与滑动窗口和基于区域提议的技术不同，YOLO 在训练和测试时都看到整幅图像，从而隐式编码了类别及其外观的上下文信息。作为顶尖检测方法之一的 Fast R-CNN [14]，常把图像中的背景区域误判为物体，因为它看不到更大的上下文。相比 Fast R-CNN，YOLO 的背景错误不到其一半。

第三，YOLO 学习到可泛化的目标表征。当在自然图像上训练、在艺术品上测试时，YOLO 大幅优于 DPM 和 R-CNN 等顶尖方法。由于 YOLO 高度可泛化，因此在应用于新领域或意外输入时更不容易崩溃。

YOLO 在精度上仍落后于最先进的检测系统。虽然它能快速识别图像中的物体，但难以精确定位某些物体，尤其是小物体。我们在实验部分进一步讨论这些权衡。

我们所有训练与测试代码均开源，并提供多种预训练模型下载。

---

## 2 统一检测

我们将目标检测的各个独立组件统一到单个神经网络中。我们的网络利用整幅图像的特征来预测每一个边界框，并同时预测一幅图像中所有类别的全部边界框。这意味着网络会对整幅图像及其中所有物体进行全局推理。YOLO 的设计使得端到端训练与实时速度成为可能，同时保持较高的平均精度。

我们的系统将输入图像划分为 S × S 网格。若某个物体的中心落入某个网格单元，则该网格单元负责检测该物体。

每个网格单元预测 B 个边界框以及这些框的置信度分数。这些置信度反映模型对“该框是否包含物体”以及“该框预测有多准确”的把握。形式上，我们将置信度定义为：

```text
confidence = Pr(Object) × IOU^truth_pred
```

如果该单元中不存在物体，则置信度分数应为零；否则，我们希望置信度等于预测框与真实框之间的交并比（IOU）。

每个边界框由 5 个预测组成：x、y、w、h 以及 confidence。其中 (x, y) 表示框中心相对于网格单元边界的坐标；宽度和高度相对于整幅图像预测；最终的置信度预测表示预测框与任意真实框之间的 IOU。

每个网格单元还预测 C 个条件类别概率 Pr(Class_i | Object)。这些概率以“该网格单元包含物体”为条件。无论边界框数量 B 是多少，我们每个网格单元只预测一组类别概率。

测试时，我们将条件类别概率与各个框的置信度预测相乘，得到**公式 (1)**：

```text
Pr(Class_i | Object) × Pr(Object) × IOU^truth_pred
                    = Pr(Class_i) × IOU^truth_pred
```

从而得到每个框的类别相关置信度分数。这些分数同时编码了“该类别出现在框中的概率”以及“预测框与物体的贴合程度”。

**图 2：模型示意。** 我们的系统将检测建模为回归问题。它将图像划分为 S×S 网格，并对每个网格单元预测 B 个边界框、这些框的置信度以及 C 个类别概率。这些预测被编码为一个 S×S×(B·5+C) 张量。

在 Pascal VOC 上评估 YOLO 时，我们取 S=7，B=2。Pascal VOC 有 20 个标注类别，因此 C=20。最终预测是一个 7×7×30 张量。

### 2.1 网络设计

**图 3：网络结构。** 我们的检测网络包含 24 个卷积层，后接 2 个全连接层。交替使用的 1×1 卷积层用于降低前层特征空间维度。我们先在 ImageNet 分类任务上以一半分辨率（224×224 输入）预训练卷积层，再将分辨率加倍用于检测。

我们将该模型实现为卷积神经网络，并在 Pascal VOC 检测数据集上评估 [9]。网络的前部卷积层提取图像特征，全连接层则预测输出概率与坐标。

我们的网络结构受用于图像分类的 GoogLeNet 启发 [34]。网络包含 24 个卷积层，后接 2 个全连接层。我们不使用 GoogLeNet 中的 Inception 模块，而是简单采用 1×1 降维层后接 3×3 卷积层，类似于 Lin 等人的做法 [22]。完整网络见图 3。

我们还训练了一个快速版 YOLO，以挑战快速目标检测的极限。Fast YOLO 使用更少卷积层（9 层而非 24 层），且这些层中的滤波器更少。除网络规模外，YOLO 与 Fast YOLO 的训练和测试参数完全相同。

网络最终输出为 7×7×30 的预测张量。

### 2.2 训练

我们在 ImageNet 1000 类竞赛数据集上预训练卷积层 [30]。预训练时，使用图 3 中的前 20 个卷积层，后接一个平均池化层和一个全连接层。我们训练约一周，在 ImageNet 2012 验证集上获得单裁剪 top-5 准确率 88%，与 Caffe Model Zoo 中的 GoogLeNet 模型相当 [24]。所有训练与推理均使用 Darknet 框架 [26]。

随后我们将模型转换为执行检测。Ren 等人表明，在预训练网络上同时增加卷积层与全连接层可以提升性能 [29]。遵循其做法，我们增加 4 个卷积层和 2 个随机初始化的全连接层。检测通常需要更细粒度的视觉信息，因此我们将网络输入分辨率从 224×224 提升到 448×448。

最后一层同时预测类别概率与边界框坐标。我们将边界框宽高按图像宽高归一化，使其落在 0 到 1 之间；并将边界框的 x 与 y 参数化为相对某网格单元位置的偏移，因此它们也落在 0 到 1 之间。

最后一层使用线性激活函数，其余各层使用如下 leaky ReLU 激活，即**公式 (2)**：

```text
φ(x) = x          ,  if x > 0
φ(x) = 0.1 x      ,  otherwise
```

我们对模型输出优化平方和误差（sum-squared error）。之所以使用平方和误差，是因为它易于优化；但它并不完美对齐“最大化平均精度”的目标。它把定位误差与分类误差同等加权，这未必理想。此外，每张图像中许多网格单元并不包含任何物体，这会把这些单元的 “confidence” 分数推向零，常常压过含有物体的单元所产生的梯度，从而导致模型不稳定，甚至在训练早期发散。

为缓解该问题，我们增大边界框坐标预测的损失，并减小“不含物体的框”的置信度损失。为此引入两个参数 λ_coord 与 λ_noobj，并设置为：

```text
λ_coord = 5
λ_noobj = 0.5
```

平方和误差还会同等对待大框与小框中的误差。我们的误差度量应体现出：大框中的小偏差通常比小框中的小偏差影响更小。为部分解决这一问题，我们预测边界框宽高的平方根，而不是直接预测宽高本身。

YOLO 在每个网格单元预测多个边界框。训练时，我们只希望一个边界框预测器对每个物体负责。我们根据“当前与真实框 IOU 最高”的原则，把某个预测器指定为对该物体“负责”。这会促使不同边界框预测器产生分工：每个预测器更擅长预测特定尺寸、长宽比或类别的物体，从而提升总体召回率。

训练中我们优化如下多部分损失函数。为便于阅读，先写成五项之和，即**公式 (3)**：

```text
L = L_xy + L_wh + L_obj + L_noobj + L_cls
```

其中各项分别为：

**中心坐标损失**

```text
L_xy = λ_coord * Σ_{i=0}^{S^2} Σ_{j=0}^{B}  1^obj_ij * [ (x_i - x̂_i)^2 + (y_i - ŷ_i)^2 ]
```

**宽高损失（对 √w、√h 回归）**

```text
L_wh = λ_coord * Σ_{i=0}^{S^2} Σ_{j=0}^{B}  1^obj_ij * [ (√w_i - √ŵ_i)^2 + (√h_i - √ĥ_i)^2 ]
```

**有目标置信度损失**

```text
L_obj = Σ_{i=0}^{S^2} Σ_{j=0}^{B}  1^obj_ij * (C_i - Ĉ_i)^2
```

**无目标置信度损失**

```text
L_noobj = λ_noobj * Σ_{i=0}^{S^2} Σ_{j=0}^{B}  1^noobj_ij * (C_i - Ĉ_i)^2
```

**分类损失**

```text
L_cls = Σ_{i=0}^{S^2}  1^obj_i * Σ_{c ∈ classes} (p_i(c) - p̂_i(c))^2
```

其中，1^obj_i 表示物体是否出现在第 i 个网格单元中；1^obj_ij 表示第 i 个单元中的第 j 个边界框预测器对该预测“负责”。

注意：损失函数仅在该网格单元存在物体时惩罚分类误差（这也对应前文所述条件类别概率）；并且仅当某个预测器对真实框“负责”（即在该单元所有预测器中 IOU 最高）时，才惩罚边界框坐标误差。

我们在 Pascal VOC 2007 与 2012 的训练与验证集上大约训练 135 个 epoch。在 2012 测试时，训练还会加入 VOC 2007 测试数据。训练全程使用 batch size 64，动量 0.9，衰减 0.0005。

学习率日程如下：最初若干 epoch 将学习率从 10^(-3) 缓慢提升到 10^(-2)。若一开始就使用高学习率，模型常因梯度不稳定而发散。随后以 10^(-2) 训练 75 个 epoch，再以 10^(-3) 训练 30 个 epoch，最后以 10^(-4) 训练 30 个 epoch。

为避免过拟合，我们使用 dropout 与大量数据增强。在第一个全连接层后加入比率为 0.5 的 dropout，以防止层间共适应 [18]。数据增强方面，引入最高达原图尺寸 20\% 的随机缩放与平移；并在 HSV 颜色空间中将曝光与饱和度随机调整到最高 1.5 倍。

### 2.3 推理

与训练一样，对测试图像预测检测结果只需一次网络前向。在 Pascal VOC 上，网络每张图像预测 98 个边界框及每个框的类别概率。由于只需单次网络评估，YOLO 在测试时极快，这与基于分类器的方法不同。

网格设计强制边界框预测具有空间多样性。通常可以清楚地判断物体落入哪个网格单元，网络也往往为每个物体只预测一个框。然而，某些大物体或靠近多个单元边界的物体，可能被多个单元较好定位。可用非极大值抑制（NMS）处理这些重复检测。虽然它对性能的关键程度不如在 R-CNN 或 DPM 中那么高，但 NMS 仍可带来约 2\%–3\% 的 mAP 提升。

### 2.4 YOLO 的局限

YOLO 对边界框预测施加了很强的空间约束：每个网格单元只预测两个框，且只能拥有一个类别。这种空间约束限制了模型可预测的邻近物体数量。我们的模型难以处理成群出现的小物体，例如鸟群。

由于模型从数据中学习预测边界框，它对具有新颖或异常长宽比、构型的物体泛化较差。此外，由于架构从输入图像开始经历多次下采样，用于预测边界框的特征相对粗糙。

最后，尽管我们在近似检测性能的损失函数上训练，该损失对小框与大框中的误差仍处理得不够理想。大框中的小误差通常无伤大雅，但小框中的小误差会对 IOU 产生大得多的影响。我们的主要误差来源是不正确的定位。

---

## 3 与其他检测系统的比较

目标检测是计算机视觉的核心问题。检测流水线通常先从输入图像提取一组鲁棒特征（Haar [25]、SIFT [23]、HOG [4]、卷积特征 [6]），再在特征空间中使用分类器 [36, 21, 13, 10] 或定位器 [1, 32] 识别物体。这些分类器或定位器要么以滑动窗口方式在整幅图像上运行，要么在图像的某些区域子集上运行 [35, 15, 39]。我们将 YOLO 与若干顶尖检测框架进行比较，突出关键异同。

**可变形部件模型（DPM）。** DPM 使用滑动窗口进行目标检测 [10]。它采用彼此割裂的流水线来提取静态特征、分类区域、对高分区域预测边界框等。我们的系统用单个卷积神经网络取代所有这些分散部件。网络同时完成特征提取、边界框预测、非极大值抑制与上下文推理。特征不再是静态的，而是在线训练并针对检测任务优化。统一架构使得模型比 DPM 更快、更准确。

**R-CNN。** R-CNN 及其变体使用区域提议而非滑动窗口来寻找图像中的物体。Selective Search [35] 生成潜在边界框，卷积网络提取特征，SVM 给框打分，线性模型调整边界框，非极大值抑制消除重复检测。复杂流水线的每个阶段都必须独立精细调参，最终系统非常慢，测试时每张图像超过 40 秒 [14]。

YOLO 与 R-CNN 有一些相似之处：每个网格单元提出潜在边界框，并用卷积特征对这些框打分。但我们的系统对网格单元提议施加空间约束，有助于缓解同一物体的多重检测；同时提议的边界框数量也少得多——每张图像仅 98 个，而 Selective Search 约 2000 个。最终，我们的系统将这些独立组件合并为单一、联合优化的模型。

**其他快速检测器。** Fast R-CNN 与 Faster R-CNN 通过共享计算，并用神经网络代替 Selective Search 来生成区域，从而加速 R-CNN 框架 [14][28]。尽管它们在速度与精度上优于 R-CNN，但仍未达到实时性能。

许多研究致力于加速 DPM 流水线 [31][38][5]：加速 HOG 计算、使用级联、把计算推到 GPU 等。然而，真正达到实时的只有 30Hz DPM [31]。

YOLO 并不试图优化大型检测流水线中的各个组件，而是直接抛弃流水线，从设计上就追求快速。

面向人脸或行人等单一类别的检测器可以高度优化，因为它们需要处理的变化更少 [37]。YOLO 则是通用检测器，学习同时检测多种物体。

**Deep MultiBox。** 与 R-CNN 不同，Szegedy 等人训练卷积神经网络来预测感兴趣区域 [8]，而不是使用 Selective Search。MultiBox 也可通过把置信度预测替换为单一类别预测来做单目标检测。但 MultiBox 无法完成通用目标检测，且仍只是更大检测流水线中的一块，还需要进一步对图像块分类。YOLO 与 MultiBox 都用卷积网络在图像中预测边界框，但 YOLO 是完整的检测系统。

**OverFeat。** Sermanet 等人训练卷积神经网络执行定位，再将该定位器改造为检测 [32]。OverFeat 能高效进行滑动窗口检测，但仍然是割裂系统。它优化的是定位而非检测性能。与 DPM 类似，定位器做预测时只看到局部信息，无法对全局上下文推理，因此需要大量后处理才能得到一致的检测结果。

**MultiGrasp。** 我们的工作在设计上与 Redmon 等人的抓取检测工作相似 [27]。我们基于网格的边界框预测思路来自 MultiGrasp 对抓取的回归系统。但抓取检测比目标检测简单得多：MultiGrasp 只需为包含单个物体的图像预测一个可抓取区域，不必估计物体大小、位置或边界，也不必预测其类别，只需找到适合抓取的区域。而 YOLO 要为图像中多个类别的多个物体同时预测边界框与类别概率。

---

## 4 实验

我们首先在 Pascal VOC 2007 上将 YOLO 与其他实时检测系统比较。为理解 YOLO 与 R-CNN 变体的差异，我们分析 YOLO 与 Fast R-CNN（R-CNN 中表现最好的版本之一）[14] 在 VOC 2007 上的错误。基于不同的错误分布，我们证明可用 YOLO 对 Fast R-CNN 检测结果重新打分，以减少背景假阳性，从而显著提升性能。我们也给出 VOC 2012 结果，并与当时最先进方法比较 mAP。最后，我们在两个艺术品数据集上表明，YOLO 对新领域的泛化优于其他检测器。

### 4.1 与其他实时系统的比较

许多目标检测研究聚焦于加速标准检测流水线 [5][38][31][14][17][28]。然而，真正做出实时（≥ 30 FPS）检测系统的，只有 Sadeghi 等人 [31]。我们将 YOLO 与其 GPU 版 DPM（30Hz 或 100Hz）比较。对尚未达到实时门槛的其他方法，我们也比较相对 mAP 与速度，以考察目标检测中可用的精度–性能权衡。

**表 1：Pascal VOC 2007 上的实时系统。** 比较快速检测器的性能与速度。Fast YOLO 是 Pascal VOC 检测记录中最快的检测器，且仍比任何其他实时检测器准确约两倍。YOLO 比快速版高约 10 mAP，同时速度仍明显高于实时要求。

| 实时检测器 | 训练数据 | mAP | FPS |
|---|---|---:|---:|
| 100Hz DPM [31] | 2007 | 16.0 | 100 |
| 30Hz DPM [31] | 2007 | 26.1 | 30 |
| Fast YOLO | 2007+2012 | 52.7 | 155 |
| YOLO | 2007+2012 | 63.4 | 45 |
| **低于实时** |  |  |  |
| Fastest DPM [38] | 2007 | 30.4 | 15 |
| R-CNN Minus R [20] | 2007 | 53.5 | 6 |
| Fast R-CNN [14] | 2007+2012 | 70.0 | 0.5 |
| Faster R-CNN VGG-16 [28] | 2007+2012 | 73.2 | 7 |
| Faster R-CNN ZF [28] | 2007+2012 | 62.1 | 18 |
| YOLO VGG-16 | 2007+2012 | 66.4 | 21 |

就我们所知，Fast YOLO 是 Pascal 上最快的目标检测方法，也是现存最快的目标检测器。其 mAP 为 52.7\%，超过此前实时检测工作两倍以上。YOLO 将 mAP 推到 63.4\%，同时仍保持实时性能。

我们也用 VGG-16 训练了 YOLO。该模型更准，但明显更慢。它便于与依赖 VGG-16 的其他检测系统比较；但由于慢于实时，后文仍聚焦我们的更快模型。

Fastest DPM 在不明显牺牲 mAP 的情况下有效加速了 DPM，但仍以约 2 倍差距未能达到实时 [38]；并且受 DPM 相对较低检测精度限制，难以与神经网络方法相比。

R-CNN Minus R 用静态边界框提议替换 Selective Search [20]。它比 R-CNN 快很多，但仍达不到实时，且因缺少优质提议而精度显著下降。

Fast R-CNN 加速了 R-CNN 的分类阶段，但仍依赖 Selective Search，后者生成边界框提议大约需要每图 2 秒。因此它 mAP 很高，但 0.5 FPS 远非实时。

最近的 Faster R-CNN 用神经网络代替 Selective Search 来提议边界框，类似于 Szegedy 等人的做法 [8]。在我们的测试中，其最准模型达到 7 FPS，较小、较不准的模型达到 18 FPS。VGG-16 版 Faster R-CNN 比 YOLO 高约 10 mAP，但也慢约 6 倍；Zeiler-Fergus 版 Faster R-CNN 只比 YOLO 慢约 2.5 倍，但精度更低。

### 4.2 VOC 2007 错误分析

为进一步考察 YOLO 与最先进检测器的差异，我们对 VOC 2007 结果做详细分解。我们将 YOLO 与 Fast R-CNN 比较，因为 Fast R-CNN 是 Pascal 上表现最好的检测器之一，且其检测结果公开可得。

我们使用 Hoiem 等人的方法与工具 [19]。对每个类别，在测试时查看该类别的前 N 个预测。每个预测要么正确，要么按错误类型分类：

- **Correct（正确）：** 类别正确且 IOU > 0.5
- **Localization（定位错误）：** 类别正确，且 0.1 < IOU < 0.5
- **Similar（相似类错误）：** 类别相似，且 IOU > 0.1
- **Other（其他错误）：** 类别错误，且 IOU > 0.1
- **Background（背景错误）：** 对任意物体都有 IOU < 0.1

**图 4：错误分析：Fast R-CNN vs. YOLO。** 这些图显示了各类别 top-N 检测中定位错误与背景错误的比例（N = 该类别物体数量）。

YOLO 难以正确定位物体。定位错误在 YOLO 的错误中占比超过其余所有来源之和。Fast R-CNN 的定位错误少得多，但背景错误多得多：其 top 检测中有 13.6\% 是不含任何物体的假阳性。Fast R-CNN 预测背景检测的可能性几乎是 YOLO 的 3 倍。

### 4.3 结合 Fast R-CNN 与 YOLO

YOLO 的背景错误远少于 Fast R-CNN。用 YOLO 消除 Fast R-CNN 中的背景检测，可显著提升性能。对 R-CNN 预测的每个边界框，我们检查 YOLO 是否预测了相似框；若是，则依据 YOLO 预测概率以及两框重叠程度，对该预测给予提升。

**表 2：VOC 2007 上的模型组合实验。** 我们考察将不同模型与最佳 Fast R-CNN 结合的效果。其他版本 Fast R-CNN 只会带来很小收益，而 YOLO 带来显著提升。

| 模型 | mAP | 组合后 | 增益 |
|---|---:|---:|---:|
| Fast R-CNN | 71.8 | - | - |
| Fast R-CNN (2007 data) | 66.9 | 72.4 | 0.6 |
| Fast R-CNN (VGG-M) | 59.2 | 72.4 | 0.6 |
| Fast R-CNN (CaffeNet) | 57.1 | 72.1 | 0.3 |
| YOLO | 63.4 | 75.0 | 3.2 |

最佳 Fast R-CNN 模型在 VOC 2007 测试集上达到 71.8\% mAP；与 YOLO 结合后，mAP 提升 3.2\% 到 75.0\%。我们也尝试把顶级 Fast R-CNN 与其他几个 Fast R-CNN 版本组合，这些集成仅带来 0.3\%–0.6\% 的小幅提升，见表 2。

来自 YOLO 的提升并非简单模型集成的副产品——因为组合不同版本 Fast R-CNN 收益很小。真正原因是：YOLO 在测试时犯的错误类型不同，因而能有效提升 Fast R-CNN。

遗憾的是，该组合并不能享受 YOLO 的速度优势，因为我们分别运行每个模型再合并结果。不过由于 YOLO 很快，相对 Fast R-CNN 并不会显著增加计算时间。

### 4.4 VOC 2012 结果

在 VOC 2012 测试集上，YOLO 得到 57.9\% mAP。这低于当时最先进水平，更接近使用 VGG-16 的原始 R-CNN，见表 3。与最接近的竞争者相比，我们的系统在小物体上更吃力。在 bottle、sheep、tv/monitor 等类别上，YOLO 比 R-CNN 或 Feature Edit 低 8\%–10\%；但在 cat、train 等类别上，YOLO 表现更高。

我们的 Fast R-CNN + YOLO 组合模型属于表现最高的检测方法之一。Fast R-CNN 因与 YOLO 结合获得 2.3\% 提升，在公开排行榜上前进 5 位。

**表 3：Pascal VOC 2012 排行榜（部分）。** YOLO 与截至 2015 年 11 月 6 日的完整 comp4（允许外部数据）公开排行榜比较。给出多种检测方法的平均精度均值与逐类平均精度。YOLO 是其中唯一实时检测器。Fast R-CNN + YOLO 是得分第四高的方法，相对 Fast R-CNN 提升 2.3\%。

| VOC 2012 test | mAP | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MR_CNN_MORE_DATA [11] | 73.9 | 85.5 | 82.9 | 76.6 | 57.8 | 62.7 | 79.4 | 77.2 | 86.6 | 55.0 | 79.1 | 62.2 | 87.0 | 83.4 | 84.7 | 78.9 | 45.3 | 73.4 | 65.8 | 80.3 | 74.0 |
| HyperNet_VGG | 71.4 | 84.2 | 78.5 | 73.6 | 55.6 | 53.7 | 78.7 | 79.8 | 87.7 | 49.6 | 74.9 | 52.1 | 86.0 | 81.7 | 83.3 | 81.8 | 48.6 | 73.5 | 59.4 | 79.9 | 65.7 |
| HyperNet_SP | 71.3 | 84.1 | 78.3 | 73.3 | 55.5 | 53.6 | 78.6 | 79.6 | 87.5 | 49.5 | 74.9 | 52.1 | 85.6 | 81.6 | 83.2 | 81.6 | 48.4 | 73.2 | 59.3 | 79.7 | 65.6 |
| Fast R-CNN + YOLO | 70.7 | 83.4 | 78.5 | 73.5 | 55.8 | 43.4 | 79.1 | 73.1 | 89.4 | 49.4 | 75.5 | 57.0 | 87.5 | 80.9 | 81.0 | 74.7 | 41.8 | 71.5 | 68.5 | 82.1 | 67.2 |
| MR_CNN_S_CNN [11] | 70.7 | 85.0 | 79.6 | 71.5 | 55.3 | 57.7 | 76.0 | 73.9 | 84.6 | 50.5 | 74.3 | 61.7 | 85.5 | 79.9 | 81.7 | 76.4 | 41.0 | 69.0 | 61.2 | 77.7 | 72.1 |
| Faster R-CNN [28] | 70.4 | 84.9 | 79.8 | 74.3 | 53.9 | 49.8 | 77.5 | 75.9 | 88.5 | 45.6 | 77.1 | 55.3 | 86.9 | 81.7 | 80.9 | 79.6 | 40.1 | 72.6 | 60.9 | 81.2 | 61.5 |
| DEEP_ENS_COCO | 70.1 | 84.0 | 79.4 | 71.6 | 51.9 | 51.1 | 74.1 | 72.1 | 88.6 | 48.3 | 73.4 | 57.8 | 86.1 | 80.0 | 80.7 | 70.4 | 46.6 | 69.6 | 68.8 | 75.9 | 71.4 |
| NoC [29] | 68.8 | 82.8 | 79.0 | 71.6 | 52.3 | 53.7 | 74.1 | 69.0 | 84.9 | 46.9 | 74.3 | 53.1 | 85.0 | 81.3 | 79.5 | 72.2 | 38.9 | 72.4 | 59.5 | 76.7 | 68.1 |
| Fast R-CNN [14] | 68.4 | 82.3 | 78.4 | 70.8 | 52.3 | 38.7 | 77.8 | 71.6 | 89.3 | 44.2 | 73.0 | 55.0 | 87.5 | 80.5 | 80.8 | 72.0 | 35.1 | 68.3 | 65.7 | 80.4 | 64.2 |
| UMICH_FGS_STRUCT | 66.4 | 82.9 | 76.1 | 64.1 | 44.6 | 49.4 | 70.3 | 71.2 | 84.6 | 42.7 | 68.6 | 55.8 | 82.7 | 77.1 | 79.9 | 68.7 | 41.4 | 69.0 | 60.0 | 72.0 | 66.2 |
| NUS_NIN_C2000 [7] | 63.8 | 80.2 | 73.8 | 61.9 | 43.7 | 43.0 | 70.3 | 67.6 | 80.7 | 41.9 | 69.7 | 51.7 | 78.2 | 75.2 | 76.9 | 65.1 | 38.6 | 68.3 | 58.0 | 68.7 | 63.3 |
| BabyLearning [7] | 63.2 | 78.0 | 74.2 | 61.3 | 45.7 | 42.7 | 68.2 | 66.8 | 80.2 | 40.6 | 70.0 | 49.8 | 79.0 | 74.5 | 77.9 | 64.0 | 35.3 | 67.9 | 55.7 | 68.7 | 62.6 |
| NUS_NIN | 62.4 | 77.9 | 73.1 | 62.6 | 39.5 | 43.3 | 69.1 | 66.4 | 78.9 | 39.1 | 68.1 | 50.0 | 77.2 | 71.3 | 76.1 | 64.7 | 38.4 | 66.9 | 56.2 | 66.9 | 62.7 |
| R-CNN VGG BB [13] | 62.4 | 79.6 | 72.7 | 61.9 | 41.2 | 41.9 | 65.9 | 66.4 | 84.6 | 38.5 | 67.2 | 46.7 | 82.0 | 74.8 | 76.0 | 65.2 | 35.6 | 65.4 | 54.2 | 67.4 | 60.3 |
| R-CNN VGG [13] | 59.2 | 76.8 | 70.9 | 56.6 | 37.5 | 36.9 | 62.9 | 63.6 | 81.1 | 35.7 | 64.3 | 43.9 | 80.4 | 71.6 | 74.0 | 60.0 | 30.8 | 63.4 | 52.0 | 63.5 | 58.7 |
| **YOLO** | **57.9** | 77.0 | 67.2 | 57.7 | 38.3 | 22.7 | 68.3 | 55.9 | 81.4 | 36.2 | 60.8 | 48.5 | 77.2 | 72.3 | 71.3 | 63.5 | 28.9 | 52.2 | 54.8 | 73.9 | 50.8 |
| Feature Edit [33] | 56.3 | 74.6 | 69.1 | 54.4 | 39.1 | 33.1 | 65.2 | 62.7 | 69.7 | 30.8 | 56.0 | 44.6 | 70.0 | 64.4 | 71.1 | 60.2 | 33.3 | 61.3 | 46.4 | 61.7 | 57.8 |
| R-CNN BB [13] | 53.3 | 71.8 | 65.8 | 52.0 | 34.1 | 32.6 | 59.6 | 60.0 | 69.8 | 27.6 | 52.0 | 41.7 | 69.6 | 61.3 | 68.3 | 57.8 | 29.6 | 57.8 | 40.9 | 59.3 | 54.1 |
| SDS [16] | 50.7 | 69.7 | 58.4 | 48.5 | 28.3 | 28.8 | 61.3 | 57.5 | 70.8 | 24.1 | 50.7 | 35.9 | 64.9 | 59.1 | 65.8 | 57.1 | 26.0 | 58.8 | 38.6 | 58.9 | 50.7 |
| R-CNN [13] | 49.6 | 68.1 | 63.8 | 46.1 | 29.4 | 27.9 | 56.6 | 57.0 | 65.9 | 26.5 | 48.7 | 39.5 | 66.2 | 57.3 | 65.4 | 53.2 | 26.2 | 54.5 | 38.1 | 50.6 | 51.6 |

### 4.5 泛化性：艺术品中的人物检测

学术目标检测数据集通常从同一分布抽取训练与测试数据。在真实应用中，很难预知所有用例，测试数据可能偏离系统所见分布 [3]。我们在 Picasso Dataset [12] 与 People-Art Dataset [3] 上比较 YOLO 与其他检测系统，这两个数据集用于测试艺术品中的人物检测。

**图 5：Picasso 与 People-Art 数据集上的泛化结果。**  
(a) Picasso 数据集的精确率–召回率曲线。  
(b) VOC 2007、Picasso 与 People-Art 上的定量结果。Picasso 同时评估 AP 与最佳 F_1。

| 方法 | VOC 2007 AP | Picasso AP | Picasso Best F_1 | People-Art AP |
|---|---:|---:|---:|---:|
| YOLO | 59.2 | 53.3 | 0.590 | 45 |
| R-CNN | 54.2 | 10.4 | 0.226 | 26 |
| DPM | 43.2 | 37.8 | 0.458 | 32 |
| Poselets [2] | 36.5 | 17.8 | 0.271 | — |
| D&T [4] | — | 1.9 | 0.051 | — |

作为参考，我们给出仅在 VOC 2007 上训练时 person 类别的 VOC 2007 AP。Picasso 上模型在 VOC 2012 上训练；People-Art 上模型在 VOC 2010 上训练。

R-CNN 在 VOC 2007 上 AP 很高，但应用到艺术品时显著下降。R-CNN 使用为自然图像调优的 Selective Search 做边界框提议；而其分类器步骤只看到小区域，因此高度依赖优质提议。

DPM 应用到艺术品时 AP 保持得较好。先前工作认为，这是因为 DPM 对物体形状与布局有很强的空间模型。尽管 DPM 退化不如 R-CNN 严重，但它起步 AP 更低。

YOLO 在 VOC 2007 上表现良好，且应用到艺术品时 AP 退化少于其他方法。与 DPM 类似，YOLO 对物体大小与形状建模，也对物体间关系以及物体常见出现位置建模。艺术品与自然图像在像素层面差异很大，但在物体大小与形状层面相似，因此 YOLO 仍能预测出较好的边界框与检测结果。

**图 6：定性结果。** YOLO 在互联网上的艺术品与自然图像样本上运行。整体大多准确，尽管它会把某个人误认为飞机。

---

## 5 真实场景中的实时检测

YOLO 是快速且准确的目标检测器，非常适合计算机视觉应用。我们将 YOLO 连接到网络摄像头，并验证其在包含取帧与显示检测结果时间的情况下仍保持实时性能。

由此得到的系统具有交互性与吸引力。虽然 YOLO 逐帧处理图像，但接到摄像头后可像跟踪系统一样工作，在物体移动与外观变化时持续检测。系统演示与源代码见项目网站：http://pjreddie.com/yolo/ 。

---

## 6 结论

我们提出 YOLO，一种统一的目标检测模型。该模型构造简单，可直接在整幅图像上训练。与基于分类器的方法不同，YOLO 在直接对应检测性能的损失函数上训练，并且整个模型联合训练。

Fast YOLO 是文献中最快的通用目标检测器，而 YOLO 推动了实时目标检测的最先进水平。YOLO 对新领域也具有良好泛化能力，因此非常适合依赖快速、鲁棒目标检测的应用。

**致谢：** 本工作部分受 ONR N00014-13-1-0720、NSF IIS-1338054 以及 The Allen Distinguished Investigator Award 支持。

---

## 参考文献

[1] M. B. Blaschko and C. H. Lampert. Learning to localize objects with structured output regression. In *Computer Vision–ECCV 2008*, pages 2–15. Springer, 2008.

[2] L. Bourdev and J. Malik. Poselets: Body part detectors trained using 3d human pose annotations. In *International Conference on Computer Vision (ICCV)*, 2009.

[3] H. Cai, Q. Wu, T. Corradi, and P. Hall. The cross-depiction problem: Computer vision algorithms for recognising objects in artwork and in photographs. *arXiv preprint arXiv:1505.00110*, 2015.

[4] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In *CVPR 2005*, volume 1, pages 886–893. IEEE, 2005.

[5] T. Dean, M. Ruzon, M. Segal, J. Shlens, S. Vijayanarasimhan, J. Yagnik, et al. Fast, accurate detection of 100,000 object classes on a single machine. In *CVPR 2013*, pages 1814–1821. IEEE, 2013.

[6] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. *arXiv preprint arXiv:1310.1531*, 2013.

[7] J. Dong, Q. Chen, S. Yan, and A. Yuille. Towards unified object detection and semantic segmentation. In *Computer Vision–ECCV 2014*, pages 299–314. Springer, 2014.

[8] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. In *CVPR 2014*, pages 2155–2162. IEEE, 2014.

[9] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective. *International Journal of Computer Vision*, 111(1):98–136, Jan. 2015.

[10] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained part based models. *IEEE TPAMI*, 32(9):1627–1645, 2010.

[11] S. Gidaris and N. Komodakis. Object detection via a multi-region & semantic segmentation-aware CNN model. *CoRR*, abs/1505.01749, 2015.

[12] S. Ginosar, D. Haas, T. Brown, and J. Malik. Detecting people in cubist art. In *Computer Vision–ECCV 2014 Workshops*, pages 101–116. Springer, 2014.

[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In *CVPR 2014*, pages 580–587. IEEE, 2014.

[14] R. B. Girshick. Fast R-CNN. *CoRR*, abs/1504.08083, 2015.

[15] S. Gould, T. Gao, and D. Koller. Region-based segmentation and object detection. In *Advances in neural information processing systems*, pages 655–663, 2009.

[16] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Simultaneous detection and segmentation. In *Computer Vision–ECCV 2014*, pages 297–312. Springer, 2014.

[17] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. *arXiv preprint arXiv:1406.4729*, 2014.

[18] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. *arXiv preprint arXiv:1207.0580*, 2012.

[19] D. Hoiem, Y. Chodpathumwan, and Q. Dai. Diagnosing error in object detectors. In *Computer Vision–ECCV 2012*, pages 340–353. Springer, 2012.

[20] K. Lenc and A. Vedaldi. R-cnn minus r. *arXiv preprint arXiv:1506.06981*, 2015.

[21] R. Lienhart and J. Maydt. An extended set of haar-like features for rapid object detection. In *ICIP 2002*, volume 1, pages I–900. IEEE, 2002.

[22] M. Lin, Q. Chen, and S. Yan. Network in network. *CoRR*, abs/1312.4400, 2013.

[23] D. G. Lowe. Object recognition from local scale-invariant features. In *ICCV 1999*, volume 2, pages 1150–1157. IEEE, 1999.

[24] D. Mishkin. Models accuracy on imagenet 2012 val. https://github.com/BVLC/caffe/wiki/Models-accuracy-on-ImageNet-2012-val. Accessed: 2015-10-2.

[25] C. P. Papageorgiou, M. Oren, and T. Poggio. A general framework for object detection. In *ICCV 1998*, pages 555–562. IEEE, 1998.

[26] J. Redmon. Darknet: Open source neural networks in c. http://pjreddie.com/darknet/, 2013–2016.

[27] J. Redmon and A. Angelova. Real-time grasp detection using convolutional neural networks. *CoRR*, abs/1412.3128, 2014.

[28] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. *arXiv preprint arXiv:1506.01497*, 2015.

[29] S. Ren, K. He, R. B. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. *CoRR*, abs/1504.06066, 2015.

[30] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision (IJCV)*, 2015.

[31] M. A. Sadeghi and D. Forsyth. 30hz object detection with dpm v5. In *Computer Vision–ECCV 2014*, pages 65–79. Springer, 2014.

[32] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. *CoRR*, abs/1312.6229, 2013.

[33] Z. Shen and X. Xue. Do more dropouts in pool5 feature maps for better object detection. *arXiv preprint arXiv:1409.6911*, 2014.

[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. *CoRR*, abs/1409.4842, 2014.

[35] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. *International journal of computer vision*, 104(2):154–171, 2013.

[36] P. Viola and M. Jones. Robust real-time object detection. *International Journal of Computer Vision*, 4:34–47, 2001.

[37] P. Viola and M. J. Jones. Robust real-time face detection. *International journal of computer vision*, 57(2):137–154, 2004.

[38] J. Yan, Z. Lei, L. Wen, and S. Z. Li. The fastest deformable part model for object detection. In *CVPR 2014*, pages 2497–2504. IEEE, 2014.

[39] C. L. Zitnick and P. Dollár. Edge boxes: Locating object proposals from edges. In *Computer Vision–ECCV 2014*, pages 391–405. Springer, 2014.

---

## 附录：核心公式速查

**置信度定义**

```text
confidence = Pr(Object) × IOU^truth_pred
```

**公式 (1)：测试时类别相关分数**

```text
Pr(Class_i | Object) × Pr(Object) × IOU^truth_pred
                    = Pr(Class_i) × IOU^truth_pred
```

**公式 (2)：Leaky ReLU**

```text
φ(x) = x       ,  if x > 0
φ(x) = 0.1 x   ,  otherwise
```

**公式 (3)：总损失（分项形式）**

```text
L = L_xy + L_wh + L_obj + L_noobj + L_cls

L_xy    = λ_coord * Σ_{i=0}^{S^2} Σ_{j=0}^{B} 1^obj_ij * [(x_i - x̂_i)^2 + (y_i - ŷ_i)^2]

L_wh    = λ_coord * Σ_{i=0}^{S^2} Σ_{j=0}^{B} 1^obj_ij * [(√w_i - √ŵ_i)^2 + (√h_i - √ĥ_i)^2]

L_obj   = Σ_{i=0}^{S^2} Σ_{j=0}^{B} 1^obj_ij * (C_i - Ĉ_i)^2

L_noobj = λ_noobj * Σ_{i=0}^{S^2} Σ_{j=0}^{B} 1^noobj_ij * (C_i - Ĉ_i)^2

L_cls   = Σ_{i=0}^{S^2} 1^obj_i * Σ_{c ∈ classes} (p_i(c) - p̂_i(c))^2
```

**VOC 设定下的输出张量**

```text
S = 7,  B = 2,  C = 20   ⇒   输出维度 = 7 × 7 × 30
```
