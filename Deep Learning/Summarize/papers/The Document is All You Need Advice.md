# 修改建议
> KingXHJ

## 2023.12.05
1. 6.6节
    - 5-gram模型应该由前四个词元组成序列？还是您想表达的意思是，前四个词元+label一块输入呢
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231205-6-6-1.png)


## 2023.12.07
1. 12.1.1节
    - 这块是不是有笔误呢？
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231207-12-1-1-1.jpg)

1. 13.6.3.2节
    - 这块用词应该是没对应上滴
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231207-13-6-3-2-1.png)
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231207-13-6-3-2-2.png)

1. 15节
    - 这个位置的日期写错了
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231207-15-1.png)


## 2023.12.08
1. 9.3节
    - 当我看到这里，我会疑惑一下：什么是基础静态词向量，什么是动态调整后的词向量？我感觉可能作为初学者，突然之间很难去理解ELMO作为一个Embedding预训练模型的意义。
    ![](../pictures/The%20Document%20is%20All%20You%20Need/20231208-9-3-1.png)

    - 我推荐在第9章章首增加一个总结性的语句：
        > Word2vec和Glove通过训练后的词向量会直接变成下游任务的输入，词向量不会随着下游任务再改变，称为静态词向量；而ELMO在Word2vec或Glove训练后的静态词向量基础上，增加了Bi-LSTM(双向LSTM)模块，相当于增加了可学习参数，在新的训练中，Bi-LSTM会学习到新的参数，从而能够根据整个模型的输入，对Word2vec或Glove训练后的静态词向量进行"动态"调整，因此，从ELMO模型中输出给下游任务的词向量，是静态词向量经过Bi-LSTM调整过的动态词向量
        
        > 配图：
        ![](../pictures/The%20Document%20is%20All%20You%20Need/20231208-9-3-2.png)