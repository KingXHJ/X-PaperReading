# 论文信息
- 时间：2022
- 期刊：ACM International Conference on Multimedia
- 网络名称： 
- 意义：
- 作者：Renrui Zhang, Ziyao Zeng, Ziyu Guo, Yafeng Li
- 实验环境：
- 数据集：
- [返回上一层 README](../README.md)
# 一、解决的问题
1. CLIP对物体非常敏感，但是对概念可能理解的不一定很好
2. 对比学习的方式，不太适用于去学习一个概念
# 二、做出的创新
![Can Language Understand Depth](../pictures/Can%20Language%20Understand%20Depth/Can%20Language%20Understand%20Depth.png)
1. 和PointCLIP流程图非常像
2. 与其把深度看作一个回归问题，不如把深度看作是一个分类问题
3. 强制性把深度问题分成几个大类
    - giant
    - extremely close
    - close
    - not in distance
    - a little remote
    - far
    - unsee
    - 对应距离[1.00 1.50 2.00 2.25 2.50 2.75 3.00]
4. 文本的prompt是：This Object is [CLASS(分好的深度类)]
5. 2D图像传入，得到视觉特征
6. 视觉特征和文本做点乘，得到的相似度矩阵做softmax
7. 知道属于哪一类之后，就可以得到一张深度估计图
# 三、设计的模型

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