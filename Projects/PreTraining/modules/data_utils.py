"""
Data utilities for text preprocessing and dataset creation
"""

import torch
from torch.utils.data import Dataset
import re
from collections import Counter


class SimpleTokenizer:
    """Simple word-level tokenizer"""
    def __init__(self, vocab_size=5000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Begin of sequence
            '<EOS>': 3,  # End of sequence
        }
        
    def build_vocab(self, text):
        """Build vocabulary from text"""
        # Tokenize
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        words = text.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter by minimum frequency and take top vocab_size
        vocab_words = [word for word, count in word_counts.items() if count >= self.min_freq]
        vocab_words = sorted(vocab_words, key=lambda w: word_counts[w], reverse=True)
        vocab_words = vocab_words[:self.vocab_size - len(self.special_tokens)]
        
        # Build mappings
        self.word2idx = self.special_tokens.copy()
        for idx, word in enumerate(vocab_words, start=len(self.special_tokens)):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} tokens")
        return self
    
    def encode(self, text):
        """Convert text to token indices"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        words = text.split()
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
    
    def decode(self, indices):
        """Convert token indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)


class TextDataset(Dataset):
    """Dataset for causal language modeling"""
    def __init__(self, text_data, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize entire corpus
        self.tokens = tokenizer.encode(text_data)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):  # Overlapping sequences
            seq = self.tokens[i:i + seq_length + 1]  # +1 for target
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
        
        print(f"✓ Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        return {
            'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
            'labels': torch.tensor(seq[1:], dtype=torch.long)
        }
