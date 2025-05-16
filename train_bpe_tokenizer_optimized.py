import regex as re
from collections import defaultdict, Counter
import json
from pre_tokenize import parallel_pretokenize


def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], path: str):
    vocab_str = {i: token.hex() for i, token in vocab.items()}
    merges_str = [[a.hex(), b.hex()] for a, b in merges]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab_str, "merges": merges_str}, f, indent=2)

def train_bpe_tokenizer_optimized(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 1. 初始化 vocab（含 256 个 byte 和 special token）
    vocab = {i: bytes([i]) for i in range(256)}
    
    token_id = 256
    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    # 2. 3. 并行预分词
    special_tokens = ["<|endoftext|>"]
    words = parallel_pretokenize("data/TinyStoriesV2-GPT4-valid.txt", special_tokens)

    # 4. 构建 pair_freq 和 pair_positions
    pair_freq = Counter()
    pair_positions = defaultdict(set)  # pair -> set of (word_idx, pos)

    for word_idx, word in enumerate(words):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += 1
            pair_positions[pair].add((word_idx, i))

    # 循环 merge 最频繁的 pair，更新 vocab、words、pair_freq、pair_positions

    merges = []
    current_token_id = max(vocab.keys()) + 1

    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        # 1. 找出当前频率最高的 pair
        best_pair, _ = pair_freq.most_common(1)[0]
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[current_token_id] = new_token
        current_token_id += 1

        # 2. 找到所有受影响的位置
        affected = pair_positions[best_pair]
        new_pair_positions = defaultdict(set)

        for word_idx, pos in list(affected):
            word = words[word_idx]

            # 如果 pair 已经被改动，跳过（pos 越界或已经被合并）
            if pos >= len(word) - 1:
                continue
            if (word[pos], word[pos + 1]) != best_pair:
                continue

            # 3. 替换 pair 为新 token
            new_word = word[:pos] + (new_token,) + word[pos + 2:]
            words[word_idx] = new_word

            # 4. 移除旧 pair 的相关位置（包括前后两个位置）
            if pos > 0:
                prev = (word[pos - 1], word[pos])
                pair_freq[prev] -= 1
                pair_positions[prev].discard((word_idx, pos - 1))
            if pos + 2 < len(word):
                nxt = (word[pos + 1], word[pos + 2])
                pair_freq[nxt] -= 1
                pair_positions[nxt].discard((word_idx, pos + 1))

            # 5. 添加新合并后的相关 pair（包括前后）
            if pos > 0:
                new_prev = (new_word[pos - 1], new_word[pos])
                pair_freq[new_prev] += 1
                new_pair_positions[new_prev].add((word_idx, pos - 1))
            if pos < len(new_word) - 1:
                new_next = (new_word[pos], new_word[pos + 1])
                pair_freq[new_next] += 1
                new_pair_positions[new_next].add((word_idx, pos))

        # 6. 清除已处理的 pair
        del pair_freq[best_pair]
        del pair_positions[best_pair]

        # 7. 更新 pair_positions
        for pair, positions in new_pair_positions.items():
            pair_positions[pair].update(positions)
    return vocab, merges


if __name__ == '__main__':
    vocab, merges = train_bpe_tokenizer_optimized("data/TinyStoriesV2-GPT4-valid.txt", 400, ["<|endoftext|>", "</w>"])
    save_bpe_model(vocab, merges, "bpe_model2.json")


