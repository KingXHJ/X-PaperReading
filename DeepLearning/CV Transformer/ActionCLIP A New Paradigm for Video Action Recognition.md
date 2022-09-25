# 论文信息
- 时间：2021
- 期刊：CVPR
- 网络名称： ActionCLIP
- 意义：CLIP进入动作识别
- 作者：
- 实验环境：
- 数据集：
# 一、解决的问题
![ActionCLIP model](../pictures/ActionCLIP%20model.png)
1. 此前的做法
    - 视频经过编码之后，得到了向量
    - 过一个分类头得到了输出
    - 和标好的数据集去做ground truth做对比
    - 问题？
        - 视频理解/动作识别中，有监督学习的标签难以定义
        - 标签类别数量受到softmax可运行范围的限制
2. multimode framework
    - 视频编码器得到特征，标签当作文本，得到文本特征
    - 算文本和图像的相似度
    - 得到相似矩阵，就和我们提前定好的ground truth算一个loss
# 二、做出的创新
1. 把图像变成视频，如何与文本算相似度
2. batch比较大的时候，相似度矩阵里就会出现，同一行或者同一列里有多个正样本，这与CLIP中，相似矩阵对角线上的才是正样本的结论不一样，就不是one hot。
    - 这是一个较容易解决的问题
    - 解决方法：cross entropy loss换成KL divergence
    - 用两个分布去算相似度就好了
# 三、设计的模型
![ActionCLIP to video](../pictures/ActionCLIP%20to%20video.png)
1. prompt（文本中的含义保持不变，视觉中的prompt更像是adapter，一些effcient fune tuning的方法）
    - effcient fune tuning：设计一些小模块，在已经预训练好的参数上，训练这些小模块，让已经训练好的模型参数能够更快的适应下游任务
2. 文本方面的prompt是前缀、中间（完形填空）和后缀
    - 劈成三类，主要是为了和视觉方面对齐
3. Pre-network Prompt：Joint
    - 时间和空间上的Token都放在一起，然后扔给网络去学习
    - 把时序embedding也会加入进去，一起学
    - Pre-network：输入层面上加了一些东西
4. In-network Prompt：shift
    - shift：在特征图上做各种移动，达到更强的建模能力，且不增加更多参数，不增加建模的复杂度，zero-cost，额外开销都是0
    - shift放到了每一个ViT的模块中间，增强模型的持续建模能力，但又不引入过多的参数和计算量
5. Post-network Prompt：和CLIP4Clip一模一样
    - 把一组frame-level representations -> 一个Video representation

# 四、实验结果

## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论