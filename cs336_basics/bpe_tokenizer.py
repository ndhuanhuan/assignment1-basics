import os
from typing import Iterable, Iterator
import regex as re
from collections import defaultdict
from multiprocessing import Process, Queue
from cs336_basics.utils import find_chunk_boundaries, split_by_special_tokens

class BPETokenizer():

    def __init__(self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    
    def tokenize(self, text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
        """
        Seperating text into pretokens
        Special tokens are independent pretokens
        """
        parts = split_by_special_tokens(text, special_tokens) # parts looks like ['Hello world! ', '<|endoftext|>', ' Great!']

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
    
    def worker(self, text: str, special_tokens: list[str], q: Queue):
        """Worker pretokenizing process for multiprocessing"""
        pretokens = self.tokenize(text, special_tokens)
        q.put(pretokens)
        print("worker done")
    
    def merge(self, counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int],set[int]], pretokens: list[list[int]], max_pair: (int, int), new_index: int):
        """Merge the pairs with highest frequency and update counts, index_dict"""
        # # counts is a defaultdict(int) where keys are pairs of integers (index1, index2), value is the count of the pair
        # index_dict, key is a tuple of two integers (index1, index2), value is index of the token in all_tokens
        # pretokens is a list of lists, where each inner list is a pretoken (list of integers)
        # max_pair is a tuple of two integers (index1, index2)
        # new_index is the index of the new merged token in the vocabulary



        # index_dict, key is a tuple of two integers (index1, index2), value is index of the token in all_tokens
        # index_set is a set of indices in pretokens where max_pair occurs
        index_set = index_dict[max_pair]

        # iterate over each token that contains max_pair
        for i in index_set:
            pretoken = pretokens[i]
            new_pretoken = [] # new token after merge looks different from the original pretoken, need to create a new one

            # Store positions of max_pair for each new pretoken after merge, 
            # even in a single pretoken, there can be multiple max_pair / merges happens
            pos_list = []   
            pos = 0
            j = 0

            # Replace max_pair with new_index in each pretoken
            while j < len(pretoken):
                if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == max_pair):
                    new_pretoken.append(new_index) # result after merge, for example (A, B) -> C then only append C
                    pos_list.append(pos) # position of the new index in the pretoken
                    j += 2
                else:
                    new_pretoken.append(pretoken[j]) # cannot merge, so keep the original byte
                    j += 1
                pos += 1 # here pos is the position of the new pretoken, not the original pretoken, you don't see plus 2 case here, because we always append new_index when we merge, so pos is always incremented by 1

            # Update counts and index_dict
            for pos in pos_list:
                counts[max_pair] -= 1 # since we merged max_pair, we need to decrease its count

                if pos > 0:
                    # If true, it means two merged pairs are now adjacent (e.g., [... new_index, new_index ...])
                    # [A, B, A, B] -> [C, C], in this case we need to reduce the count of the pair (B, A)
                    # So it need to reduce reversed C in this case, which is (max_pair[1], max_pair[0])
                    if new_pretoken[pos-1] == new_index:
                        counts[(max_pair[1], max_pair[0])] -= 1
                    # else case, If the previous symbol is not the new merged symbol, you decrement the count for the pair that was previously there: (new_pretoken[pos-1], max_pair[0]).
                    else:
                        counts[(new_pretoken[pos-1], max_pair[0])] -= 1

                    counts[(new_pretoken[pos-1], new_pretoken[pos])] += 1 # a new pair is created after merge
                    index_dict[(new_pretoken[pos-1], new_pretoken[pos])].add(i) # update index_dict with the new pair

                if pos < len(new_pretoken)-1:
                    if new_pretoken[pos+1] == new_index:
                        counts[(max_pair[1], max_pair[0])] -= 1     
                    else:
                        counts[(max_pair[1], new_pretoken[pos+1])] -= 1

                    counts[(new_pretoken[pos], new_pretoken[pos+1])] += 1
                    index_dict[(new_pretoken[pos], new_pretoken[pos+1])].add(i)

            pretokens[i] = new_pretoken
    
    def train_bpe(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Args:
        #     input_path (str | os.PathLike): Path to BPE tokenizer training data.
        #     vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        #     special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        #         These strings will never be split into multiple tokens, and will always be
        #         kept as a single token. If these special tokens occur in the `input_path`,
        #         they are treated as any other string.

        # Returns:
        #     tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        #         vocab:
        #             The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
        #             to bytes (token bytes)
        #         merges:
        #             BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
        #             representing that <token1> was merged with <token2>.
        #             Merges are ordered by order of creation.
    
        # Initializations
        special_tokens = special_tokens or []
        num_merges = max(vocab_size - len(special_tokens) - 256, 0)

        chunk_list = []
        num_chunks= 4

        # Initialize vocab
        vocab = {}
        vocab = {x:bytes([x]) for x in range(0,256)}
        for i, token in enumerate(special_tokens):
            vocab[256+i] = token.encode("utf-8")
        merges = []
        
        # Note: How vocab looks like
        # vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff', 256: b'<|endoftext|>', ...}
        
        # Chunk the text file
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))
            print("boundaries are:", boundaries)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_list.append(chunk)

        print("chunk_list length is:", len(chunk_list))
        # print("chunk_list[0] is:", chunk_list[0])

        # Parallel tokenize
        pretokens_list = []
        processes = []
        q = Queue()
        for chunk in chunk_list:
            p = Process(target=self.worker, args=(chunk, special_tokens, q))
            p.start()
            processes.append(p)

        pretokens_list = [q.get() for _ in processes]

        for p in processes:
            p.join()

        pretokens = [token for tokens in pretokens_list for token in tokens]

        print("pretokens length is:", len(pretokens))
        print("pretokens[:10] is:", pretokens[:10]) # pretokens[:10] is: [b'u', b' don', b"'t", b' have', b' to', b' be', b' scared', b' of', b' the', b' loud']

        # Merging
        counts = defaultdict(int)
        index_dict = defaultdict(set)  # key is a tuple of two integers (index1, index2), value is a set of indices in all_tokens when tuple occurs

        # Iterates over every token in all_tokens, where j is the index and token is a bytes object (e.g., b'Hello').
        for j, token in enumerate(pretokens):
            # For each token (which is a bytes object), this loops over every pair of consecutive bytes in that token.
            # If token = b'cat', then zip(token, token[1:]) yields (99, 97) and (97, 116) (ASCII codes for 'c', 'a', 't').
            # token looks like b'cat', index1 = 99, index2 = 97
            for index1, index2 in zip(token, token[1:]):
                counts[index1, index2] += 1 # this is for finding largest counts later,  so we can merge the most frequent pairs
                index_dict[index1, index2].add(j) # this is for easily finding the index of this pair in pretokens, used later in merge function
        
        for i in range(num_merges):
            # Prefer lexicographically greater pair
            # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
            max_pair = max(
                counts.items(),
                key=lambda x: (
                    x[1],  
                    vocab[x[0][0]].decode("utf-8", errors="ignore"),
                    vocab[x[0][1]].decode("utf-8", errors="ignore")
                )
            )[0]

            index1, index2 = max_pair

            new_index = 256 + len(special_tokens) + i

            vocab[new_index] = vocab[index1] + vocab[index2]
            merges.append((vocab[index1], vocab[index2]))

            # counts is a defaultdict(int) where keys are pairs of integers (index1, index2), value is the count of the pair
            # index_dict, key is a tuple of two integers (index1, index2), value is index of the token in all_tokens
            # max_pair is a tuple of two integers (index1, index2)
            # new_index is the index of the new merged token in the vocabulary
            self.merge(counts, index_dict, pretokens, max_pair, new_index)

        return (vocab, merges)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges"""
        raise NotImplementedError
    
    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        byte_pretokens = self.tokenize(text, self.special_tokens, drop_special_token=False)   # list[bytes]
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = []  # list[list[int]]

        # Convert pretokens from bytes to list[int] by vocab
        for i, pretoken in enumerate(byte_pretokens):

            new_pretoken = []

            if pretoken in byte_special_tokens:
                index = vocab_reversed[pretoken]
                new_pretoken.append(index)
            else:
                for b in pretoken:
                    index = vocab_reversed[bytes([b])]
                    new_pretoken.append(index)

            pretokens.append(new_pretoken)

        # Merge
        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_pretoken = []
                new_index = vocab_reversed[merge[0] + merge[1]]
                j = 0
                while j < len(pretoken):
                    if (j < len(pretoken)-1) and ((self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1

                pretoken = new_pretoken

            pretokens[i] = pretoken

        tokens = [token for pretoken in pretokens for token in pretoken] 
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files 
        that we cannot directly load into memory.
        """
        for line in iterable:
            for idx in self.encode(line):
                yield idx


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]    # bytes
            else:
                token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 