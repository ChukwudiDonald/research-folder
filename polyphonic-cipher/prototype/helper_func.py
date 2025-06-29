import re
import torch
import string

from typing import Union, List, Optional

INPUT_VOCAB = list(string.ascii_lowercase) + ['[', ']', 'X', ' ']
TARGET_VOCAB =['a', 'b', 'e', 'g', 'i', 'j', 'k', 'o', 'p', 'q', 't', 'v', 'x', 'y', 'z']

def tokenize_with_brackets(word: str) -> list[tuple[str, str]]:
    """
    Modified implementation to return the output in the specified format,
    revealing one more cluster character at each step.
    """
    CLUSTERS = {
    'e': '[ezv]',
    'z': '[ezv]',
    'v': '[ezv]',
    'a': '[axy]',
    'x': '[axy]',
    'y': '[axy]',
    't': '[tqb]',
    'q': '[tqb]',
    'b': '[tqb]',
    'o': '[ojg]',
    'j': '[ojg]',
    'g': '[ojg]',
    'i': '[ikp]',
    'k': '[ikp]',
    'p': '[ikp]'
    }

    # First identify all cluster character positions and the characters themselves
    cluster_chars_with_pos = [(i, c) for i, c in enumerate(word) if c in CLUSTERS]
    cluster_positions = [pos for pos, char in cluster_chars_with_pos]

    results = []

    num_cluster_chars = len(cluster_chars_with_pos)

    # Iterate through each cluster character to generate the output tuples
    # i represents the index of the tuple being generated (0-indexed)
    for i in range(num_cluster_chars):
        # The revealed character for this tuple is the i-th cluster character in the word
        revealed_char = cluster_chars_with_pos[i][1]

        # Build the masked version based on the current tuple index 'i'.
        # Cluster characters with index < i are revealed, those >= i are masked.
        masked = []
        cluster_char_count = 0 # To track which cluster character we are currently processing
        for pos, char in enumerate(word):
            if pos in cluster_positions: # It's a cluster character
                if cluster_char_count < i:
                    # This cluster character should be revealed in this tuple's string
                    masked.append(char)
                else:
                    # This cluster character and subsequent ones should be masked
                    masked.append(CLUSTERS[char])
                cluster_char_count += 1 # Move to the next cluster character index
            else: # Non-cluster character
                masked.append(char)

        results.append((''.join(masked), revealed_char))

    return results

def flexible_tokenizer(
    word: Optional[str] = None,
    tokens: Optional[List[int]] = None,
    vocab: List[str] = None,
    pad_len: Optional[int] = None,
    unk_token: str = ' '
) -> Union[int, List[int], str]:
    """
    Flexible version of your original tokenizer that:
    - Takes vocabulary as input (not hardcoded)
    - Handles both tokenization and detokenization
    - Maintains all original padding behavior
    """
    # Default to your original alphabet if no vocab provided
    if vocab is None:
        raise ValueError("No vocab was provided")

    # Detokenization mode (tokens → word)
    if tokens is not None:
        return ''.join([vocab[i] if i < len(vocab) else unk_token 
                       for i in tokens]).strip()
    
    # Tokenization mode (word → tokens)
    if word is not None and pad_len is not None:
        if len(word) == 1:
            try:
                return vocab.index(word)
            except ValueError:
                return vocab.index(unk_token)

        indices = [
            vocab.index(c) if c in vocab else vocab.index(unk_token)
            for c in word.lower()
        ]
        return indices[:pad_len] + [vocab.index(unk_token)] * max(0, pad_len - len(indices))
    
    raise ValueError("Must provide either (word + pad_len) or tokens")

def predict_correct_word(model, input_word, input_vocab=INPUT_VOCAB, target_vocab=TARGET_VOCAB):
    model.eval()
    current_word = tokenize_with_brackets(input_word)[0][0]
    
    while '[' in current_word:
        # Prepare input tensor
        tokenized = flexible_tokenizer(word=current_word, pad_len=len(current_word), vocab=input_vocab)
        input_tensor = torch.tensor(tokenized, dtype=torch.long).unsqueeze(0)
        
        # Predict next letter
        with torch.no_grad():
            predicted_idx = torch.argmax(model(input_tensor), dim=-1).item()
            predicted_letter = flexible_tokenizer(tokens=[predicted_idx], vocab=target_vocab)
        
        # Replace first cluster
        current_word = re.sub(r'\[.*?\]', predicted_letter, current_word, count=1)
    
    return current_word





