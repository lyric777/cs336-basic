### 自己训练一个BPE分词器
### 使用TinyStoriesV2-GPT4-train.txt(2.3G，最后也没用），TinyStoriesV2-GPT4-valid.txt(22M)训练vocab和merge规则
按照assignment1指导
- ### Pre-tokenization
use a regex-based pre-tokenizer (used by GPT-2)
- ### Special tokens
除了<|endoftext|>，我还添加了b'</w>'
- ### Parallelizing pre-tokenization
使用提供的pretokenization_example.py规则划分chunk，multiprocessing并行处理，并且remove special tokens before pre-tokenization
- ### Optimizing the merging step
构建 pair_freq 和 pair_positions两个dict，避免每次迭代重新计算所有pair的频数
