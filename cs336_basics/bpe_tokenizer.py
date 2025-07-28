from .base_tokenizer import Tokenizer, get_stats, merge, find_chunk_boundaries

import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue
import time
import multiprocessing

class BPETokenizer(Tokenizer):

    def __init__(self):
        # super().__init__()
        print('init()') # skip original init for now
    
    def split_by_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        """
        Split on the special tokens
        example: 
            text = "Hello world! <|endoftext|> Great!" 
            special_tokens = "<|endoftext|>"
            result = ['Hello world! ', '<|endoftext|>', ' Great!']
        """
        # Sorts tokens by length (longest first). This prevents partial matches when tokens overlap (e.g., <|end|> and <|endoftext|>).
        special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
        if not special_tokens_sorted:
            parts = [text]
        else:
            # Escapes each token for safe regex use.
            # Joins them with | (regex OR), so any token can be matched.
            pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
            parts = re.split('(' + pattern + ')', text)

        return parts
    
    def tokenize(self, text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
        """
        Seperating text into pretokens
        Special tokens are independent pretokens
        """
        parts = self.split_by_special_tokens(text, special_tokens)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens_list = []
        for part in parts:
            if part in special_tokens:
                if not drop_special_token:  # Keep special tokens, otherwise ignore
                    spec_tok_bytes = part.encode('utf-8')
                    tokens_list.append([spec_tok_bytes])
            else:
                str_tokens = re.findall(PAT, part)
                part_tokens = [s.encode('utf-8') for s in str_tokens]
                tokens_list.append(part_tokens)
        tokens = [token for part_tokens in tokens_list for token in part_tokens]
        return tokens
    
    def _tokenize_helper(self, args):
        tokenizer, chunk, special_tokens, drop_special_token = args
        return tokenizer.tokenize(chunk, special_tokens, drop_special_token)
    
    def train_bpe(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
        # Initializations
        special_tokens = special_tokens or []
        num_merges = max(vocab_size - len(special_tokens) - 256, 0)
        special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}
        chunk_list = []
        num_chunks= 4
        
        # Chunk the text file
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, special_token_bytes)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_list.append(chunk)
    
        # Parallel tokenize
        args_list = [(self, chunk, special_tokens, True) for chunk in chunk_list]

        # token_lists_per_chunk is a list of lists of tokens, one per chunk
        token_lists_per_chunk = []
        with multiprocessing.Pool() as pool:
            token_lists_per_chunk = pool.map(self._tokenize_helper, args_list)
        
        all_tokens = [token for token_list in token_lists_per_chunk for token in token_list]

        # Merging
        counts = defaultdict(int)
        index_dict = defaultdict(set)  # Store pretoken location for each pair


    def train(self, text, vocab_size, verbose=False):
        return None

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids