# 论文信息
- 时间：2020
- 期刊：ECCV
- 网络名称： DETR(detection Transformer)
- 意义：Transformer
- 作者：Nicolas Carion*, Francisco Massa*, Gabriel Synnaeve, Nicolas Usunier,Alexander Kirillov, and Sergey Zagoruyko; Facebook AI
- 实验环境：
- 数据集：
# 一、解决的问题
1. 目标检测很少有端到端的方法，大多都需要一个后处理操作(nms, non-maximum suppersion 非极大值抑制)。
    - 不论是proposal based方法(RCNN)，还是anchor based方法(Yolo)，还是non-anchor based方法，最后都会生成很多预测框。如何去除这些冗余的框，就是nms要做的事情。但是有了nms的存在，这个模型的调参就比较复杂，而且即使训练好了一个模型，部署起来也是非常困难。因为nms这个操作，不是所有硬件都支持的。所以简单的端到端系统，是大家一直以来梦寐以求的。
2. 想让目标检测和图像分类一样，用简单优雅的框架给他做出来，而不是像之前的框架一样，需要很多的人工干预，需要很多的先验知识，需要很多的库，或者普通硬件不支持的一些算子
# 二、做出的创新
1. end-to-end
2. Transformer
3. 不需要proposal，不需要anchor，利用Transformer的全局建模能力，把目标检测堪称一个集合预测的问题，并且全局建模能力保证DETR不会输出那么多冗余的框
4. **把目标检测堪称一个集合预测的问题**
5. 提出了一个新的目标函数，去用二分图匹配，输出一组独一无二的预测
6. 第二个贡献就是使用了Transformer encoder和decoder的架构
    - 解码器有另外一个输入->learned object query，和全局图像信息结合，通过不停的去做注意力这种操作，从而能够让模型直接输出最后的一组预测框，而且还是（in-parallel）
7. ***DETR其实是一个框架***
# 三、设计的模型
1. DETR流程
    ![DETR works](../pictures/DETR%20works.png)
    - 先用CNN去抽取特征
    - 然后拉直，送入Transformer的encoder（学习全局信息）和decoder里面去
    - decoder输出框
    - 输出的框和ground Truth的框去做一个匹配，在匹配的框里去算目标检测的loss（推理不需要，用一个阈值去卡置信度即可->0.7）
    - 没画object query
2. Object detection set prediction loss
    - DETR任何时候都会输出N个预测框（文章中 $N=100$ ）
    ![DETR loss](../pictures/DETR%20loss.png)
    
    $$\mathbf{L}_{Hungarian}(y,\hat{y})=\sum_{i=1}^{N}[-log\hat{p}_ {\hat{\sigma}(i)}(c_i)+\mathbb{1}_ {\{c_i\neq \varnothing \}}\mathbf{L}_{box}(b_i,\hat{b}_ {\hat{\sigma}}(i))]$$
    - 分类的loss和匹配框的loss
    - 一定要得到一对一的结果
    - 先算最优匹配，再在上面算loss

3. DETR结构：
    ![DETR architecture](../pictures/DETR%20architecture.png)
    ![DETR arch analyse](../pictures/DETR%20arch%20analyse.png)


# 四、实验结果 

## 1、比之前模型的优势
1. 检测大物体更有优势，小物体Faster RCNN更强。可能是因为全局建模的缘故
## 2、有优势的原因
1. 简单性：
    - 不需要特殊的库，只需要库和硬件支持CNN和Transformer即可
2. 性能上：
    - 在COCO数据集上，DETR和一个训练的非常好的Faster-RCNN的基线网络效果差不多
## 3、改进空间
- DETR训练的太慢
# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码
```
1 import torch
2 from torch import nn
3 from torchvision.models import resnet50
4
5 class DETR(nn.Module):
6
7 def __init__(self, num_classes, hidden_dim, nheads,
8 num_encoder_layers, num_decoder_layers):
9 super().__init__()
10 # We take only convolutional layers from ResNet-50 model
11 self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
12 self.conv = nn.Conv2d(2048, hidden_dim, 1)
13 self.transformer = nn.Transformer(hidden_dim, nheads,
14 num_encoder_layers, num_decoder_layers)
15 self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
16 self.linear_bbox = nn.Linear(hidden_dim, 4)
17 self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
18 self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
19 self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
20
21 def forward(self, inputs):
22 x = self.backbone(inputs)
23 h = self.conv(x)
24 H, W = h.shape[-2:]
25 pos = torch.cat([
26 self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
27 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
28 ], dim=-1).flatten(0, 1).unsqueeze(1)
29 h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
30 self.query_pos.unsqueeze(1))
31 return self.linear_class(h), self.linear_bbox(h).sigmoid()
32
33 detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
34 detr.eval()
35 inputs = torch.randn(1, 3, 800, 1200)
36 logits, bboxes = detr(inputs)
```
# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
