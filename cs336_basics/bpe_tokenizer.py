import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue
import time
import multiprocessing

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        parts = re.split('(' + pattern + ')', text)

    return parts


class BPETokenizer():

    def __init__(self):
        # super().__init__()
        print('init()') # skip original init for now

    
    def tokenize(self, text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
        """
        Seperating text into pretokens
        Special tokens are independent pretokens
        """
        parts = split_by_special_tokens(text, special_tokens)

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
            pos_list = []   # Store positions of max_pair for each new pretoken after merge
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
                pos += 1

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
        
        # Chunk the text file
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_list.append(chunk)

        print("chunk_list length is:", len(chunk_list))

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

        # Merging
        counts = defaultdict(int)
        index_dict = defaultdict(set)  # key is a tuple of two integers (index1, index2), value is a set of indices in all_tokens when tuple occurs

        # Iterates over every token in all_tokens, where j is the index and token is a bytes object (e.g., b'Hello').
        for j, token in enumerate(pretokens):
            # For each token (which is a bytes object), this loops over every pair of consecutive bytes in that token.
            # If token = b'cat', then zip(token, token[1:]) yields (99, 97) and (97, 116) (ASCII codes for 'c', 'a', 't').
            for index1, index2 in zip(token, token[1:]):
                counts[index1, index2] += 1
                index_dict[index1, index2].add(j)
        
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

    def decode(self, ids):
        return None
    
    def encode(self, text):
        return None