# test_bpe_tokenizer.py

from cs336_basics.bpe_tokenizer import BPETokenizer

def main():
    file_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train_bpe(file_path, vocab_size, special_tokens)
    print("Vocab size:", len(vocab))
    print("First 10 merges:", merges[:10])

if __name__ == "__main__":
    main()