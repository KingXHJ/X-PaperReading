# 论文信息
- 时间：2021
- 期刊：CVPR
- 网络名称： CLIP4Clip
- 意义：CLIP在视频领域的应用
- 作者：Huaishao Luo1∗, Lei Ji2, Ming Zhong3, Yang Chen3, Wen Lei3, Nan Duan2, Tianrui Li1； 1Southwest Jiaotong University, Chengdu, China, 2Microsoft Research Asia, Beijing, China, 3Microsoft STCA, Beijing, China
- 实验环境：
- 数据集：MSR-VTT, MSVC, LSMDC, ActivityNet, and DiDe
# 一、解决的问题

# 二、做出的创新

# 三、设计的模型
![CLIP4Clip framework](../pictures/CLIP4Clip%20framework.png)
1. CLIP天生就适合做retrive的任务，就是在算图像和文本之间的相似性
2. CLIP是双塔结构，图像和文本编码器分开的，最后做一步点乘，就能得到相似性
3. CLIP4Clip
    - 文本是一句话，Tokenize之后，扔给了Text Encoder，就是一个Transformer
    - 得到一个cls token（整体句子的一个表达）
    - 视频是多帧，经过ViT得到了多个cls Token
    - 问题：一个文本cls对应多个视频（/图片）cls？
        - 文章是Empirical Study，尝试了3个方法
4. 三种计算相似度方法
    1. parameter-free type
        - 不需要学习任何东西，直接取平均值
        - 没有考虑时序特性
        - 局限性很强，但是应用很广
    2. sequential type
        - 时序建模（late fusion），只考虑最后的融合
        - 用Transformer（需要加上position）或者LSTM
    3. tight type
        - early fusion
        - 最开始就进行融合
# 四、实验结果

## 1、比之前模型的优势

## 2、有优势的原因
![CLIP4Clip result](../pictures/CLIP4Clip%20result.png)
- CLIP预训练，且迁移性非常好
- 数据集较少，平均法更好
    - 简单
    - 快速
- tight type需要更多的数据集，目前效果不太好
- **learning rate敏感，非常重要的参数**
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