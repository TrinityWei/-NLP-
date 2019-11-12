# Word2vec“无痛”实践指南

*by Wei mengxin*




## 一、引言
本实例主要介绍的是选取wiki中文语料，并使用python完成Word2vec模型构建的实践过程，不包含原理部分，旨在一步一步的了解自然语言处理的基本方法和步骤。

 - gensim库，可以帮助我们“**无痛**”训练word2vec，是目前最简单有效的工具。[gensim库官方地址](https://radimrehurek.com/gensim/index.html)
    - 安装gensim之前要确保已经安装了C编译器（MacOS、Linux系统自带无需担心，Windows系统就看各位的造化了），不然就没办法编译word2vec，gensim比起NumPy实现的简单版本速度提升了70%! 
 - 本来准备选用腾讯AI实验室的中文语料库：[腾讯AI实验室中文语料库](hhttps://ai.tencent.com/ailab/nlp/embedding.html)
    - 奈何这个语料库太大，压缩版就有6G以上，解压后估计15G左右，不是一般的电脑能处理的，故而放弃。退而求其次，选择wiki的中文语料库，压缩版1.74G，解压后7.2G，大家的电脑勉强可用。 
- 基于各位同学的计算机性能和计算机知识水平，所有程序运行结束估计需要1.5--2小时
---

## 二、实践过程
### 2.1 获取Wiki中文语料库
请自行前往wiki官网下载中文语料，下载完成后会得到命名为zhwiki-latest-pages-articles.xml.bz2的文件，大小约为1.74G，是一个巨大的XML文件。
 - [Wiki中文语料库下载](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)
- 我们首先需要将XML文件转换为txt格式，这个txt文件中每一行就是语料库中的一篇中文文章
- 使用gensim.corpora中的WikiCorpus函数可以很方便的处理维基百科的数据，我们定义一个函数：

```python
import time  # 插入time库，记录程序运行时间
from gensim.corpora import WikiCorpus  # 


def xml_to_txt(input_path, output_file):
    '''该函数负责将下载的XML文件转换成txt文件'''
    start = time.time()
    i = 0
    space = " "
    output = open(output_file, 'w')
    wiki = WikiCorpus(input_path, lemmatize=False, dictionary=[])  # gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts():  # 通过get_texts将维基里的每篇文章转换位1行txt文本，并且去掉了标点符号等内容
        output.write(space.join(text) + "\n")
        i = i+1
        if i % 2000 == 0:  # 每转换好2000个文章，提示我们并记录运行时间
            now = time.time()
            print("Finished Saved %s articles and Time consuming: %f 秒" % (i, (now - start)))
    output.close()
    end = time.time()
    print("Finished Saved %s articles and Time consuming: %f 秒" % (i, (now - start)))
    return output_file


inputs = "/Users/mengxinwei/Downloads/zhwiki-latest-pages-articles.xml.bz2" # 这是下载好的wiki语料库的路径，请改成你自己电脑的存放路径
outputs = "/Users/mengxinwei/Downloads/zhwiki.txt" # 这是保存转成txt文件的路径

get_text = xml_to_txt(inputs, outputs)  # 最后，我们调用该函数

```

上述code运行结束，就会在outputs = "/Users/mengxinwei/Downloads/zhwiki.txt"路径下生成txt文件，约1.19G大小
以下是最后运行结果：
```
Finished Saved 324000 articles and Time consuming: 653.084448 秒
Finished Saved 326000 articles and Time consuming: 657.721831 秒
Finished Saved 328000 articles and Time consuming: 661.348681 秒
Finished Saved 330000 articles and Time consuming: 665.327459 秒
Finished Saved 332000 articles and Time consuming: 669.330893 秒
Finished Saved 334000 articles and Time consuming: 674.031454 秒
Finished Saved 336000 articles and Time consuming: 678.053614 秒
Finished Saved 338000 articles and Time consuming: 682.249996 秒
Finished Saved 338005 articles and Time consuming: 682.249996 秒

 # 根据不同计算机硬件，耗时会不同！
 # 可以看到wiki中文语料库包含33.8万的中文文章，我这里耗时约11分钟
```


---



### 2.2 繁体中文转换成简体中文
Wiki中文语料中包含了很多繁体字，需要转成简体字再进行处理。

这里我们需要使用一个新的库：**zhconv**

安装**zhconv**的方法，在cmd窗口（Windows系统）或终端（MacOS）中输入下条语句：

```
pip3 install zhconv
```

具体实施code如下：

```python
import zhconv
import time

def convert_to_simpChinese(input_tradChinese_path, out_simpChinese_path):
    '''该函数主要负责将繁体中文转换成简体中文'''
    start = time.time()
    with open(input_tradChinese_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        print("读取完毕！")
        with open(out_simpChinese_path, 'w', encoding='UTF-8') as f1:
            print("正在转换....")
            f1.write(zhconv.convert(content, 'zh-cn'))
    end = time.time()
    print("Finished converted and Time consuming: %f 秒" % (end - start))


inputs = "/Users/mengxinwei/Downloads/zhwiki.txt"  # 这是包含繁体中文的txt格式的语料库
outputs = "/Users/mengxinwei/Downloads/zhwiki_simp.txt"  # 这是转成简体txt文件的保存路径

convert_to_simpChinese(inputs, outputs)

```
运行结果如下：

```python
读取完毕！
正在转换....
Finished converted and Time consuming: 482.337443 秒
```
得到我们自己定义的简体中文txt文件——zhwiki_simp.txt，大小约1.19G

---

### 2.3 中文分词
然后我们采用结巴分词对字体简化后的wiki中文语料数据集进行分词，在执行代码前需要安装jieba模块。
 - 安装**zhconv**的方法，在cmd窗口（Windows系统）或终端（MacOS）中输入下条语句：

```
pip3 install jieba
```

> 由于此语料已经去除了标点符号，因此在分词程序中无需进行清洗操作，可直接分词。若是自己采集的数据还需进行标点符号去除和去除停用词的操作。


```python
import jieba
import codecs
import time


def Chinese_word_segmentation(input_file_path, output_file_path):
    '''该函数负责完成中文分词工作'''
    start = time.time()
    file = codecs.open(input_file_path, 'r', encoding='utf8')
    output = codecs.open(output_file_path, 'w', encoding='utf8')
    print('open files.')
    line_num = 1
    line = file.readline()
    while line:
        if line_num % 1000 == 0:
            print('Now processing ', line_num, ' article')
        seg_list = jieba.cut(line, cut_all=False)
        line_seg = ' '.join(seg_list)
        output.writelines(line_seg)
        line_num = line_num + 1
        line = file.readline()
    end = time.time()
    print("Chinese words segmentation completed and Time consuming: %f 秒" % (end - start))
    f.close()
    output.close()


input_file = "/Users/mengxinwei/Downloads/zhwiki_simp.txt"
output_file = "/Users/mengxinwei/Downloads/zhwiki_simp_seg.txt"

seg = Chinese_word_segmentation(input_file, output_file)

```
最终分词结果如下：

```
Now processing  332000  article
Now processing  333000  article
Now processing  334000  article
Now processing  335000  article
Now processing  336000  article
Now processing  337000  article
Now processing  338000  article
Chinese words segmentation completed and Time consuming: 2062.340123 秒
```
分词确实相当耗时！

---

### 2.4 Word2Vec模型训练
分好词的文档即可进行word2vec词向量模型的训练了。文档较大，至少需要8G内存（请关闭不必要的程序，以避免程序报错）。分词非常耗时，要有心理准备！为了记录数据训练的过程，我们引入logging库，来记录程序运行的情况

> 我自己的16G内存Macbook可完成训练

具体Python代码实现如下所示:

```python
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告

import logging
import os.path
import sys
import multiprocessing
import time

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    start = time.time()
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inputs为输入语料, output_model为输出模型, output_vector为word2vec的vector格式的模型
    inputs = '/Users/mengxinwei/Downloads/zhwiki_simp_seg.txt'
    output_model = '/Users/mengxinwei/Downloads/wiki.zh.text.model'
    output_vector = '/Users/mengxinwei/Downloads/wiki.zh.text.vector'

    # 训练skip-gram模型
    # sg为训练方法，1为Skip-gram,0为COBW
    # size为词向量长度
    # window为预测前后长度
    # min_count表示最小词频数，小于设定的值会被舍弃
    # workers=multiprocessing.cpu_count()表示调用多线程cpu_count()内为空表示，该计算机最大线程数
    model = Word2Vec(LineSentence(inputs), sg=1, size=300, window=4, min_count=5,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(output_model)
    model.wv.save_word2vec_format(output_vector, binary=False)
    end = time.time()
    print("Data set training completed and Time consuming %s 秒" % (end - start))


```
训练结束后（耗时约25分钟），输出如下：

```
2019-05-17 22:10:34,499: INFO: worker thread finished; awaiting finish of 2 more threads
2019-05-17 22:10:34,504: INFO: worker thread finished; awaiting finish of 1 more threads
2019-05-17 22:10:34,507: INFO: worker thread finished; awaiting finish of 0 more threads
2019-05-17 22:10:34,507: INFO: EPOCH - 5 : training on 200524590 raw words (187046184 effective words) took 219.4s, 852596 effective words/s
2019-05-17 22:10:34,508: INFO: training on a 1002622950 raw words (935226709 effective words) took 1195.2s, 782480 effective words/s
2019-05-17 22:10:34,533: INFO: saving Word2Vec object under /Users/mengxinwei/Downloads/wiki.zh.text.model, separately None
2019-05-17 22:10:34,570: INFO: storing np array 'vectors' to /Users/mengxinwei/Downloads/wiki.zh.text.model.wv.vectors.npy
2019-05-17 22:10:38,156: INFO: not storing attribute vectors_norm
2019-05-17 22:10:38,183: INFO: storing np array 'syn1neg' to /Users/mengxinwei/Downloads/wiki.zh.text.model.trainables.syn1neg.npy
2019-05-17 22:10:40,620: INFO: not storing attribute cum_table
2019-05-17 22:10:40,620: WARNING: this function is deprecated, use smart_open.open instead
2019-05-17 22:10:42,847: INFO: saved /Users/mengxinwei/Downloads/wiki.zh.text.model
2019-05-17 22:10:42,851: INFO: storing 828236x300 projection weights into /Users/mengxinwei/Downloads/wiki.zh.text.vector
2019-05-17 22:10:42,851: WARNING: this function is deprecated, use smart_open.open instead
Data set training completed and Time consuming 1441.4469871520996 秒
```

此时在你自己定义的路径会生成2个重要文件：
- wiki.zh.text.model  本质上就是我们需要的**Look-up table**,大小约54M
- wiki.zh.text.vector 储存所有词条的300维向量，大小约**4G**

---

### 2.5 模型训练结果测试
现在我们来看一下，我们折腾了半天，这个模型训练的结果。
#### 2.5.1 相似度测试


```python
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import gensim


if __name__ == '__main__':
    model = gensim.models.Word2Vec.load("/Users/mengxinwei/Downloads/wiki.zh.text.model")
    
    word = model.most_similar("股票")
    for t in word:
        print(t[0], t[1])

```
我们测试词条“股票”，即可得到以下，最高相似度的词条：

```
2019-05-17 22:18:15,137: INFO: loading Word2Vec object from /Users/mengxinwei/Downloads/wiki.zh.text.model
2019-05-17 22:18:15,143: WARNING: this function is deprecated, use smart_open.open instead
2019-05-17 22:18:20,653: INFO: loading wv recursively from /Users/mengxinwei/Downloads/wiki.zh.text.model.wv.* with mmap=None
2019-05-17 22:18:20,653: INFO: loading vectors from /Users/mengxinwei/Downloads/wiki.zh.text.model.wv.vectors.npy with mmap=None
2019-05-17 22:18:21,371: INFO: setting ignored attribute vectors_norm to None
2019-05-17 22:18:21,371: INFO: loading vocabulary recursively from /Users/mengxinwei/Downloads/wiki.zh.text.model.vocabulary.* with mmap=None
2019-05-17 22:18:21,371: INFO: loading trainables recursively from /Users/mengxinwei/Downloads/wiki.zh.text.model.trainables.* with mmap=None
2019-05-17 22:18:21,371: INFO: loading syn1neg from /Users/mengxinwei/Downloads/wiki.zh.text.model.trainables.syn1neg.npy with mmap=None
2019-05-17 22:18:22,496: INFO: setting ignored attribute cum_table to None
2019-05-17 22:18:22,496: INFO: loaded /Users/mengxinwei/Downloads/wiki.zh.text.model
/Applications/PyCharm.app/Contents/helpers/pydev/pydevconsole.py:9: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
  
2019-05-17 22:18:25,535: INFO: precomputing L2-norms of word weight vectors
普通股 0.7055182456970215
公司股票 0.6935920119285583
期货 0.6876181364059448
股价 0.6789758801460266
期权 0.6697543859481812
股票市场 0.6662347316741943
权证 0.6575611233711243
新股 0.6539663076400757
优先股 0.6510415077209473
证券 0.6440172791481018
```
#### 2.5.2 词条计算

```
    # 其意义是计算一个词，使得该词的向量v(d)与v(a="皇后")-v(c="皇帝")+v(b="国王")最近
    
    word = model.most_similar(positive=['皇帝','国王'],negative=['皇后'])
    for t in word:
        print(t[0], t[1])
```
其结果如下：

```python
/Applications/PyCharm.app/Contents/helpers/pydev/pydevconsole.py:1: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
 
君主 0.5656882524490356
教皇 0.5607707500457764
统治者 0.5555317401885986
一世 0.5488530397415161
教宗 0.544150710105896
三世 0.526948094367981
二世 0.5262294411659241
摄政王 0.5237387418746948
哈里发 0.5209563970565796
王位 0.5121110081672668
```
#### 2.5.3 找出一组词条中特殊的一个

```python
    print(model.doesnt_match(u'飞机 苹果 香蕉 火龙果 橘子 榴莲'.split()))
    print(model.doesnt_match(u'太后 皇贵妃 贵妃 妃子 贵人'.split()))
```
输出结果：

```python
/Applications/PyCharm.app/Contents/helpers/pydev/pydevconsole.py:1: DeprecationWarning: Call to deprecated `doesnt_match` (Method will be removed in 4.0.0, use self.wv.doesnt_match() instead).
'''
飞机
太后
```

#### 2.5.4 计算任意2个词条的相似度

```
    print(model.similarity(u'经济', u'金融'))
    print(model.similarity(u'经济', u'央行'))
    print(model.similarity(u'经济', u'银行'))
    print(model.similarity(u'经济', u'计算机'))
    print(model.similarity(u'经济', u'睡觉'))
    print(model.similarity(u'经济', u'逛街'))
```
输出如下：

```python
/Applications/PyCharm.app/Contents/helpers/pydev/pydevconsole.py:1: DeprecationWarning: Call to deprecated `similarity` (Method will be removed in 4.0.0, use self.wv.similarity() instead).

0.54626226
0.27370563
0.17740533
-0.057285856
-0.036419544
```



---

by wei mengxin

End . 2019.5.17 夜
