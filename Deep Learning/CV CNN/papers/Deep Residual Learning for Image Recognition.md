# 论文信息
- 时间：2015
- 期刊：CVPR
- 网络名称： ResNet
- 意义：构建深层网络都要有的残差连接
- 作者：Kaiming He；Xiangyu Zhang；Shaoqing Ren；Jian Sun-Microsoft Research
- 实验环境：
- 数据集：ILSVRC & COCO 2015 competitions
- [返回上一层 README](../README.md)
- 
# 一、解决的问题
1. >Is learning better networks as easy as stacking more layers?
发现网络越深，反倒准确率下降了，梯度爆炸或者消失
2. 团队认为这并不是过拟合导致的，并且指出并不是所有的系统都同样易于优化->因为训练精度和测试精度都很差
# 二、做出的创新
1. 一个残差学习框架，用于简化网络的训练，这些网络比以前使用的网络要深得多
# 三、设计的模型
1. >a depth of up to 152 layers—8×deeper than VGG nets
2. >a deep residual learning framework

![ResidualLearning](../pictures/Residual%20Learning/Residual_learning.png)

- 我认为就是由于网络深度增加，信息到高维后变得稀疏，关系丢失，原始信息零散化

3. 假定某神经网络的输入是x，期望输出是H(x)，如果我们直接把输入x传到输出作为初始结果，那么此时我们需要学习的目标就是F(x) = H(x) - x。一个残差学习单元（Residual Unit）如下图所示，ResNet相当于将学习目标改变了，不再是学习一个完整的输出H(x)，只是输出和输入的差别H(x) - x，即残差。
4. ResNet has no hidden fc layers
5. ***用1X1的卷积做投影，保证通道数匹配，也可以降维***

# 四、实验结果
## 1、比之前模型的优势
1. >Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases; 
2. >Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks
3. >Our 152-layer residual net is the deepest network ever presented on ImageNet, while still having lower complexity than VGG nets 

## 2、有优势的原因
- 加入了残差网络，修复了深度模型准确率下降的问题，提高了准确率
- residual在梯度方面保持的很好，在BP的时候，越往前传，小的梯度相乘就消失了，但是residual的存在，保证可以在BP的时候有个较大的项，与乘了很多小的梯度相加
- 内在模型复杂度降低了，就不容易过拟合了
## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
- 通过残差思想解决了
## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题
- 通过实验，ResNet随着网络层不断的加深，模型的准确率先是不断的提高，达到最大值（准确率饱和），然后随着网络深度的继续增加，模型准确率毫无征兆的出现大幅度的降低。这个现象与“越深的网络准确率越高”的信念显然是矛盾的、冲突的。ResNet团队把这一现象称为“退化（Degradation）”。ResNet团队把退化现象通过恒等映射的方式进行优化。
# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
- 不一定是要原创一些东西，而是把前人做过的工作巧妙地结合起来，就很出色
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
