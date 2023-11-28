# 论文信息
- 时间：2021
- 期刊：ICCV
- 网络名称： Swin Transformer
- 意义：多层次的Vision Transformer
- 作者：Ze Liu†*, Yutong Lin†*, Yue Cao*, Han Hu*‡, Yixuan Wei†, Zheng Zhang, Stephen Lin, Baining Guo; Microsoft Research Asia
- 实验环境：
- 数据集：COCO、ADE20K

[ppt](../ppt/Swin%20Transformer/SwinTRM.pdf)

[python code](../code/Swin%20Transformer/swin_transformer.py)

# 一、解决的问题
1. 告诉大家，Transformer可以被应用于视觉的方方面面，作为通用的骨干网络
2. 使用Transformer有尺度的问题
3. 图像的分辨率太大了，需要减少序列程度
# 二、做出的创新
1. 用了移动窗口的层级Transformer，其实就是让ViT能像卷积神经网络一样，做层级的特征提取，***提供多尺寸特征***
2. 引入***移动窗口***，减小分辨率，提供各个尺度的分辨率，还保证了层级之间的关系
3. 类似池化的方法，`patch merging`
# 三、设计的模型

![Swin Transformer Hierarchical](../pictures/Swin%20Transformer/Swin%20Transformer%20Hierarchical.png)

- 类似池化的方法，`patch merging`

![Swin Transfomer Shift Window](../pictures/Swin%20Transformer/Swin%20Transformer%20Shift%20Window.png)

- 窗口整体向右下角移动，重新分割，达到了Transformer学习上下文，提供了学习窗口之间的联系

![Swin Transformer](../pictures/Swin%20Transformer/Swin%20Transformer.png)

![Swin Tranfomer Patch merging](../pictures/Swin%20Transformer/Swin%20Transfomer%20Patch%20merging.png)

- LN -> 一个窗口的自注意力 -> LN -> MLP -> shift window -> 基于移动窗口的自注意力 -> LN -> MLP
- 都是偶数的block，因为Transfomer block都是两个两个组合的
> - 主要贡献：
  1. 全局自注意力计算太贵了，用是window（固定7X7）
  2. window节省了内存和计算量，但是窗口之间没有联系了，因此发明了shift window，解决窗口和窗口之间的联系

# 四、实验结果

## 1、比之前模型的优势
- 部分能达到全方位碾压的结果
## 2、有优势的原因
- 移动窗口和相对位置编码在分类任务提升不大
- 对目标检测和语义分割等下游任务非常明显
## 3、改进空间
- shift window需要应用于NLP
# 五、结论

## 1、模型是否解决了目标问题
- NLP和CV大一统
- 多模态
## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题
- 虽然移动窗口解决了窗口联系，但是窗口移动后，大小不同，且数量改变，提高了计算复杂度\

![Swin Tranfomer shift window mask](../pictures/Swin%20Transformer/Swin%20Transfomer%20shift%20window%20mask.png)

- 通过循环移位，补充空缺位置；用完再还原回去

> 问题是，位置上的联系混论了，于是进行掩码解决

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
