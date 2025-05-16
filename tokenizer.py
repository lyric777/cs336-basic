import regex as re
import json
from collections.abc import Iterable, Iterator

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Merge map for fast lookup
        self.merge_map = {pair: pair[0] + pair[1] for pair in self.merges}

        # Reverse vocab: bytes -> ID
        self.byte_to_id = {v: k for k, v in self.vocab.items()}

        # ID to bytes map for decode
        self.id_to_byte = self.vocab

        # Ensure special tokens are in vocab
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id
                self.id_to_byte[new_id] = token_bytes

    @classmethod
    def from_files(cls, model_path: str, special_tokens: list[str] | None = None):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = self.pattern.findall(text)
        output_ids = []

        for token in tokens:
            word = [bytes([b]) for b in token.encode("utf-8")] + [b'</w>']
            i = 0
            while i < len(word) - 1:
                pair = (word[i], word[i+1])
                if pair in self.merge_map:
                    merged = self.merge_map[pair]
                    word[i:i+2] = [merged]
                    i = max(i - 1, 0)
                else:
                    i += 1

            for token_bytes in word:
                token_id = self.byte_to_id.get(token_bytes)
                if token_id is not None:
                    output_ids.append(token_id)
                else:
                    raise ValueError(f"Unknown token: {token_bytes}")

        return output_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b""
        for token_id in ids:
            token_bytes = self.id_to_byte.get(token_id)
            if token_bytes is None:
                raise ValueError(f"Unknown token ID: {token_id}")
            byte_seq += token_bytes

        # Remove end-of-word markers
        byte_seq = byte_seq.replace(b'</w>', b'')

        try:
            return byte_seq.decode("utf-8")
        except UnicodeDecodeError:
            return byte_seq.decode("utf-8", errors="replace")


if __name__ == '__main__':
    tokenizer = Tokenizer.from_files("bpe_model2.json", special_tokens=["<|endoftext|>"])
    text = """u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
        <|endoftext|>
        Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
        Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
        They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
        <|endoftext|>"""
    ids = tokenizer.encode(text)
    print("Encoded:", ids)

    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)

    # Lazy encode
    """with open("data.txt", "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            print(token_id)"""