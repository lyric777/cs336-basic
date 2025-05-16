from collections import defaultdict, Counter
import json
from pre_tokenize import parallel_pretokenize

def get_pair_frequencies(words):
        pair_freq = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += 1
        return pair_freq


def merge_pair(words, pair_to_merge):
    merged_words = []

    pair_first, pair_second = pair_to_merge
    new_token = pair_first + pair_second  # b'he'

    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair_first and word[i + 1] == pair_second:
                new_word.append(new_token)
                i += 2  # skip next token
            else:
                new_word.append(word[i])
                i += 1
        merged_words.append(tuple(new_word))

    return merged_words


def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], path: str):
    vocab_str = {i: token.hex() for i, token in vocab.items()}
    merges_str = [[a.hex(), b.hex()] for a, b in merges]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab_str, "merges": merges_str}, f, indent=2)

def load_bpe_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
    return vocab, merges

def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 读取原始数据并编码为字节（UTF-8）
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 初始化词表：所有单字节（0~255）+ 特殊 token（转成 bytes）
    vocab = {i: bytes([i]) for i in range(256)}
    token_id = 256

    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")  # 加入特殊 token
        token_id += 1

    # 将字节序列划分为“词”（word），每个词就是一个句子或空格间隔
    # 注意：这是 byte-level，不是 str.split()
    # words = [tuple(w) + (b'</w>',) for w in byte_data.split(b' ')]  # 添加结束标记，这句不对，会造成[(101, b'</w>'),] tuple里有int和byte
    """ byte_data = text.encode("utf-8")  # 原先的方法，只考虑空格
    words = [
        tuple([bytes([b]) for b in w]) + (b'</w>',)
        for w in byte_data.split(b' ')
    ] """
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # pre-tokenization的要求，GPT2的方法
    pattern = re.compile(PAT)
    tokens = pattern.findall(text)
    words = [tuple(bytes([t]) for t in token.encode("utf-8")) + (b'</w>',) for token in tokens if token]
    
    pair_freq = get_pair_frequencies(words)

    """ for pair, freq in pair_freq.most_common(10):
        print(pair, freq) """   


    # 初始化 merges 列表（空） BPE 训练逻辑…
    merges = []
    token_id = max(vocab.keys()) + 1  # 下一个可用 token ID

    while len(vocab) < vocab_size:
        pair_freq = get_pair_frequencies(words)
        if not pair_freq:
            break  # 没有可以合并的 pair 了

        # 找出频率最高的 pair
        most_common_pair, _ = pair_freq.most_common(1)[0]

        # 合并这个 pair
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab[token_id] = new_token
        merges.append(most_common_pair)

        # 更新 words
        words = merge_pair(words, most_common_pair)
        token_id += 1

    return vocab, merges






def train_bpe_tokenizer_from_words(words: list[tuple[bytes]], vocab_size: int, special_tokens: list[str]) -> tuple[dict, list]:
    vocab = {i: bytes([i]) for i in range(256)}
    token_id = 256

    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    # 初始化 merges 列表
    merges = []
    token_id = max(vocab.keys()) + 1

    while len(vocab) < vocab_size:
        pair_freq = get_pair_frequencies(words)
        if not pair_freq:
            break

        most_common_pair, _ = pair_freq.most_common(1)[0]
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab[token_id] = new_token
        merges.append(most_common_pair)

        words = merge_pair(words, most_common_pair)
        token_id += 1

    return vocab, merges

""" vocab, merges = train_bpe_tokenizer("data/TinyStoriesV2-GPT4-valid.txt", 259, ["<|endoftext|>"])
save_bpe_model(vocab, merges, "bpe_model.json") """

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    words = parallel_pretokenize("data/TinyStoriesV2-GPT4-valid.txt", special_tokens)

    vocab_size = 300  # 可自定义
    vocab, merges = train_bpe_tokenizer_from_words(words, vocab_size, special_tokens)
    save_bpe_model(vocab, merges, "bpe_model.json")

    # 打印几个 vocab 和 merge 看看
    print("\nSample vocab items:")
    for i, (token_id, token_bytes) in enumerate(vocab.items()):
        print(token_id, token_bytes)
        if i >= 9: break

    print("\nSample merges:")
    for merge in merges[:10]:
        print(merge)


