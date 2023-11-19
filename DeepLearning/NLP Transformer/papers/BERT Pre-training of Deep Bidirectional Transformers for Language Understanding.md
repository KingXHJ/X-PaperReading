# 论文信息
- 时间：2018
- 期刊：ACL
- 网络名称： BERT
- 意义：Transformer一统NLP的开始
- 作者：Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova; Google AI Language
- 实验环境：
- 数据集：

[ppt](../ppt/models_by_DASOU/BERT/BERT.pptx)

[code anlysis](../ppt/models_by_DASOU/BERT/BERTCode.pdf)

[python code](../ppt/models_by_DASOU/BERT/BERT.py)


# 一、解决的问题
1. 在NLP中，BERT之前，始终没有一个大的、深的神经网络可以在很大的数据集上训练好之后，帮助一大批的NLP任务
2. 参考ELMo（芝麻街里的人物），他们凑了一个BERT（芝麻街里的人物），Bideirection Encoder Representations from Transformer
3. 单向分析句子是非常不合理的，也不符合部分事实
# 二、做出的创新
1. 与ELMo和GPT不同，相比于GPT只用左侧信息，BERT用了右侧和左侧的信息去做预测，所以它是双向的；ELMo用的是基于RNN的架构，BERT用的是Transformer，所以ELMo做下游任务的时候，要对结构进行一些调整，而BERT就相对简单很多，只需要改对象就可以了
2. 受1953年一篇close的论文影响，使用了带掩码的语言模型，随机将句子中的词进行掩码
4. 还训练了输入两个句子，让模型判断两个句子是否相关，或者是随机的两个句子
5. BERT是第一个基于微调的模型，在词源层面和句子层面任务都很好
6. 重点就是把前人的工作，拓展到了深的、双向结构上
# 三、设计的模型
1. 预训练是在没有标号的数据集上进行训练
2. 微调是用预训练得到的参数做初始化，再用有标号的数据集，让所有参数都参与调整

- 预训练和微调相同的部分
    ![BERT pre fine](../pictures/BERT%20pre%20fine.png)

    1. 多层的、双向的Transformer
    2. 作者调整的参数：
        - L：Transformer blocks
        - H：hidden size
        - A：self-attention heads

    3. 计算参数

    ![BERT param](../pictures/BERT%20param.png)

    - A * 64 = H

    4. 每个序列的第一个词都是[CLS]，BERT希望最后用[CLS]代表整个句子的信息
    5. 区分两个句子：
        - 用[SEP]分开

    ![BERT embedding](../pictures/BERT%20embedding.png)

    6. BERT Embedding
        - 词向量
        - 第一个句子\第二个句子
        - 位置向量

- 预训练和微调不一样的部分：

    1. 预训练有15%的掩码（mask），微调就没有mask，这会出现看到的词数量不一样的问题
    2. 解决办法是：
        - 15%的mask中，80%使用[MASK]替换了
        - 10%随机换了一个词根
        - 10%什么都不变，只是标记这个词要做预测
        - 这个比率是用实验得到的
    3. 预测句子之间的关系
        - 50%正例，50%负例

3. BERT跟一些基于编码器解码器的模型不同的地方是，self-attention能够再两端之间相互能够看到彼此；但是一般模型的编码器看不到解码器的东西。BERT的问题是因为这个优势，就难以像Transformer去做机器翻译

# 四、实验结果
1. GLUE：句子层面的任务，[CLS]拿出来，学习一个输出层
2. SQuAD：给一个问题，在文章中找答案
3. SWAG：判断两个句子之间的关系
## 1、比之前模型的优势

## 2、有优势的原因
- 消融实验：去掉任何一个都会有效果的打折
    - 去掉下一个句子的预测
    - 只是从左看到右，而不是用带有掩码的语言模型
    - 或者加入BiLSTM

- 大模型效果更好
## 3、改进空间
- 机器翻译会打折
- 文本生成也不会那么好了
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