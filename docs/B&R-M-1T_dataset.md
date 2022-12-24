一带一路多语言数据集-1TB （B&R-M-1T）

> **背景描述**

中国已经同140个国家和31个国际组织签署共建“一带一路”合作文件，其中共涉及12语系，28语族，132种语言，语言多样性是造成语言障碍的主要因素。由于语种使用人口、地理分布的不均衡、社会信息化水平的差异以及语料收集渠道的隔离，使得国内优秀的机器翻译产品在小语种翻译效果上也难以发挥优势。
近年，随着NLP预训练模型和多语言模型的兴起，形成了一个拥有强大潜力的范式：使用多种语言的数据组合训练一个模型以实现多语种间的互译。因为只需要为所有语言对构建和使用一个模型，而不是为每个语言对构建一个模型，它极大地简化了系统的开发和部署。其次，可以通过利用多语种的双语语料和单语语料在不同资源量级不同语言的数据中转移知识的能力来提高低资源语种的翻译质量。


> **数据说明**
每个文件为该语种单或双语抽样语料，目前包含52种语种数据，所有语料来自于PanGu-Alpha中文语料、CC-100、CCMatrix、UN Parallel Corpus、WMT等经过规则过滤、全局精确和模糊去重、双语字符对齐过滤等清洗流程得到
文件名带"corpus"字段均为双语语料对，"\t"分割。否者为单语语料句
举例如下：
zh-en_corpus.txt
```
而且，俩人都从事石油行业相关工作。	They both work in the oil field business.

Mary Ng （伍凤仪），加拿大华裔联邦国会议员，几天前刚刚被委任加拿大小型企业及出口促进部部长。	Congratulations are in order for Mary Ng, who was recently announced as Canada’s Minister of Small Business and Export Promotion.

第三，无功绩。	third, with no success.

从Scrapy 1.0开始，所有发行版都应被视为可投入生产.	Starting with Scrapy 1.0, all releases should be considered production-ready.

其他提议包括:鼓励当地金融服务提供者参与当地、地区和全球电子金融平台,帮助中小企业学习如何在线支付或接受付款,获得电子信贷和进入其他电子金融安排,鼓励国际金融机构将中小企业获得电子金融机会作为企业战略中的一个重要部分	other proposals included : encouraging local financial service providers ' participation in local , regional and global e-finance platforms , assisting smes to learn how to pay or get paid online , getting e-credits and entering into other e-finance arrangements ; and encouraging ifis to make smes access to e-finance an important part of their strategies .

```
en.txt
```
If you believe that your privacy rights have been violated, you may file a complaint in writing with the Facility or with the Office for Civil Rights (“OCR”) in the U.S. Department of Health and Human Services. We will not retaliate against you for filing a complaint.

Keeping on top of your vital signs is important to monitoring your overall health. From digital bathroom scales to body fat scales, the best measuring scales not only provide accurate readings, but are sturdy and easy to operate. If you suffer from low vision, select a scale with an easy-to-read backlit LCD display.

Ahhh...! If walking into a Chipotle and inhaling the spicy goodness of their burritos isn't one of the best smells in the world, we don't know what is!

12. The method according to claim 10, wherein the pressing of the housing elements (5) takes place by means of the shaping process "vertically pressing".

```

各语种单双语数据容量如下所示，单语文件名为对应{key}.txt，双语文件名为对应{key}_corpus.txt，单位GB。
```
mono = {'es': 21.64, 'he': 10.79, 'cs': 5.96, 'vi': 27.05, 'lt': 4.51, 'sr': 3.24, 'lo': 0.203, 'fr': 19.11, 'hi': 5.4,
        'uz': 0.246, 'lv': 2.97, 'uk': 28.14, 'mn': 0.861, 'ms': 2.43, 'az': 1.77, 'ps': 0.22, 'hr': 6.76, 'sl': 3.53, 
        'nl': 12.27, 'el': 14.77, 'ar': 11.52, 'th': 23.14, 'bg': 18.77, 'ne': 1.11, 'pt': 19.83, 'bs': 0.03, 'id': 44.05,
        'ur': 1.79, 'fa': 31.08, 'zh': 115.06, 'km': 0.467, 'ru': 115.0, 'en': 117.28, 'tr': 8.69, 'hu': 19.38, 
        'ro': 20.02, 'be': 1.41, 'bn': 1.95, 'ta': 3.48, 'et': 2.29, 'sk': 7.74, 'de': 27.21, 'hy': 1.74, 'kk': 1.99, 
        'ko': 18.39, 'my': 0.592, 'si': 1.13, 'sq': 1.65, 'pl': 18.53, 'mk': 1.71, 'ka': 3.1, 'tl': 1.0}
zh_corpus = {'zh-ro': 0.53, 'zh-ar': 2.32, 'zh-vi': 0.592, 'zh-ta': 0.066, 'zh-lt': 0.137, 'zh-ru': 3.21, 'zh-tl': 0.015, 
             'zh-bg': 0.573, 'zh-el': 0.541, 'zh-hi': 0.172, 'zh-tr': 0.544, 'zh-th': 0.159, 'zh-hu': 0.358, 'zh-pl': 0.606, 
             'zh-bs': 0.068, 'zh-nl': 0.372, 'zh-ur': 0.016, 'zh-en': 4.73, 'zh-bn': 0.111, 'zh-uk': 0.123, 'zh-sr': 0.152, 
             'zh-de': 0.991, 'zh-ko': 0.202, 'zh-es': 1.56, 'zh-pt': 0.69, 'zh-sl': 0.257, 'zh-fa': 0.269, 'zh-hr': 0.245, 
             'zh-fr': 2.63, 'zh-id': 0.298, 'zh-ms': 0.074, 'zh-he': 0.351, 'zh-sk': 0.177, 'zh-et': 0.218, 'zh-mk': 0.112, 'zh-cs': 0.479}
en_corpus = {'en-az': 0.055, 'en-sq': 1.21, 'en-lt': 1.4, 'en-sr': 1.46, 'en-el': 4.34, 'en-hr': 1.07, 'en-sk': 1.93, 
             'en-ur': 0.339, 'en-ar': 4.17, 'en-de': 17.45, 'en-fa': 1.75, 'en-ta': 0.467, 'en-sl': 1.54, 'en-tr': 2.57, 
             'en-si': 0.339, 'en-he': 1.41, 'en-ko': 0.687, 'en-ru': 11.97, 'en-hi': 1.35, 'en-vi': 3.79, 'en-tl': 0.135, 
             'en-ro': 3.77, 'en-mk': 0.892, 'en-pt': 12.99, 'en-fr': 23.66, 'en-ne': 0.04, 'en-uk': 1.48, 'en-et': 1.23, 
             'en-bn': 0.677, 'en-bg': 3.7, 'en-be': 0.053, 'en-hu': 2.28, 'en-ms': 0.543, 'en-nl': 6.63, 'en-cs': 3.43, 
             'en-es': 32.47, 'en-pl': 4.89, 'en-id': 5.03}
ar_corpus = {'ar-tl': 0.059, 'ar-hy': 0.015, 'ar-kk': 0.008, 'ar-si': 0.135, 'ar-ur': 0.119, 'ar-ne': 0.382, 
             'ar-bn': 0.236, 'ar-ko': 0.173, 'ar-az': 0.023, 'ar-uk': 0.218, 'ar-my': 0.003, 'ar-ka': 0.006, 'ar-ms': 0.155, 
             'ar-km': 0.008, 'ar-ta': 0.155, 'ar-uz': 0.09}
ru_corpus = {'ne-ru': 0.014, 'ko-ru': 0.231, 'mk-ru': 0.487, 'ru-uk': 3.46, 'ru-si': 0.093, 'ru-ta': 0.138, 
             'ru-tl': 0.035, 'be-ru': 0.44, 'ru-ur': 0.104, 'ms-ru': 0.16, 'bn-ru': 0.213}
other_corpus = {'be-cs': 0.112, 'he-ne': 0.169, 'az-kk': 0.004, 'ka-tr': 0.008, 'bn-ur': 0.056, 'si-ur': 0.024, 
             'be-sl': 0.036, 'be-uk': 0.072, 'hy-km': 0.002, 'hy-my': 0.001, 'mk-uk': 0.152, 'ne-ur': 0.002, 'fa-uz': 0.093, 
             'bg-mk': 0.404, 'hr-mk': 0.148, 'fa-ur': 0.138, 'ms-tl': 0.009, 'ko-vi': 0.161, 'my-tr': 0.007, 'id-tl': 0.098, 
             'hi-ta': 0.041, 'be-hr': 0.006, 'az-my': 0.001, 'ne-ta': 0.002, 'tr-ur': 0.19, 'hi-si': 0.04, 'be-pl': 0.186, 
             'hi-ur': 0.089, 'bn-hi': 0.176, 'id-ms': 0.379, 'tr-uz': 0.283, 'ka-vi': 0.01, 'az-id': 0.065, 'id-si': 0.133, 
             'si-ta': 0.02, 'ne-si': 0.001, 'km-tr': 0.013, 'sk-uk': 0.351, 'hy-ka': 0.002, 'az-km': 0.003, 'ne-tr': 0.544, 
             'kk-tr': 0.011, 'mk-sk': 0.148, 'az-tr': 0.083, 'bn-si': 0.04, 'kk-vi': 0.007, 'bn-ta': 0.039, 'ka-kk': 0.001, 
             'bn-sr': 0.072, 'sr-uk': 0.301, 'bn-ne': 0.003, 'ta-ur': 0.036, 'be-bg': 0.078, 'az-vi': 0.016}
```


语种列表：
<div align=center>
<img src="../examples/mPanGu/doc/languages.png" width="800"/><br/>
</div>

> **数据来源**

鹏城实验室-智能部-高效能云计算所-分布式计算研究室

> **问题描述**

多语言机器翻译方法主要集中在以英语为中心的方向上，而非英语方向还比较滞后，本数据集提供清洗后的以中文为中心的多语言机器翻译的单双语高质量语料，希望在一带一路低资源小语种方向提供给大家一个研究探索的数据范例。


