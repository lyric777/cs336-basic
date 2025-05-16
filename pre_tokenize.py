import os
import regex as re
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT)

def pretokenize_chunk(text: str, special_tokens: List[str]) -> List[Tuple[bytes]]:
    # 先去除所有 special token
    pattern_str = "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(pattern_str, text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_regex = re.compile(PAT)

    words = []
    for chunk in chunks:
        tokens = token_regex.findall(chunk)
        for token in tokens:
            byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8")) + (b'</w>',)
            words.append(byte_tuple)
    return words


def parallel_pretokenize(input_path: str, special_tokens: List[str]) -> List[Tuple[bytes]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_count(), special_tokens[0].encode("utf-8"))

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    with Pool() as pool:
        results = pool.starmap(pretokenize_chunk, [(chunk, special_tokens) for chunk in chunks])

    # Flatten
    all_words = [word for chunk_words in results for word in chunk_words]
    return all_words
