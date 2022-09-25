# 论文信息
- 时间：2021
- 期刊：ICLR
- 网络名称： 
- 意义：CLIP用回到VIsion Language
- 作者：Sheng Shen∗†, Liunian Harold Li∗‡, Hao Tan◦, Mohit Bansal◦,Anna Rohrbach†, Kai-Wei Chang‡, Zhewei Yao† and Kurt Keutzer†； †University of California, Berkeley, ‡University of California, Los Angeles ◦University of North Carolina at Chapel Hill
- 实验环境：
- 数据集：
# 一、解决的问题

# 二、做出的创新
1.  是第一个做大规模empircal study
    - 拿CLIP预训练模型，当作视觉编码器的初始化参数
    - 再在下游的各种Vision Language上，去做Fine tune
    - 看看CLIP这个初始化参数到底好不好使
2. 就是把视觉编码器换成了CLIP的模型，其他的都是之前的方法
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