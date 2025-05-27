# Context Cipher: Concept Note

*A summary of the idea, its motivation, and proof-of-concept results.*

---

## Motivation

Modern encryption standards—such as **AES**, **RSA**, and others—reliably perform mathematical operations on the numeric representations of alphabetic characters. The most widely used mapping system, **ASCII** (American Standard Code for Information Interchange), facilitates seamless global data exchange with clarity and efficiency. However, ASCII was designed for standardization, **not** cryptography.  
If the core objective of encryption is to render plain information unrecognizable, why rely on a mapping system intended for the opposite purpose?

### Key Limitations of Conventional Mapping

- **Fixed one-to-one character-to-code correspondence** (e.g., `a = 97`, `b = 98`, `c = 99`)—this simplicity enables frequency-analysis attacks.  
- **Predictable ciphertext structure**—common di-grams and tri-grams persist, leaving patterns exploitable by statistical methods.

This project explores an alternative approach: **distorting the foundational mapping layer *before* applying standard encryption**.  
By leveraging **deep-learning sequence models**, we automate reconstruction of the original message, enhancing security through obfuscation before encryption.

---

## The Core Idea

Instead of using ASCII's bijective character encoding, we first apply a non-injective (many-to-one) mapping from alphabetic characters to numeric representations, creating intentional overlaps in the coding space. This means the same numeric value can decode to different alphabetic characters, with the correct interpretation determined by its position or neighboring symbols. We then process these ambiguous numeric representations through conventional ciphers (AES, RSA, etc.). The result is ciphertext that - even before encryption - bears no statistical resemblance to natural language text, as it eliminates predictable character distributions and n-gram patterns.


### Custom Letter-to-Index Mapping

```python
letter_to_index_custom = {
    'a': 0,  'b': 1,  'c': 2,  'd': 3,  'e': 4,
    'f': 5,  'g': 6,  'h': 7,  'i': 8,  'j': 6,
    'k': 8,  'l': 9,  'm': 10, 'n': 11, 'o': 6,
    'p': 8,  'q': 1,  'r': 12, 's': 13, 't': 1,
    'u': 14, 'v': 4,  'w': 15, 'x': 0,  'y': 0,
    'z': 4
}
```

#### Visualizing the Mapping

```mermaid
xychart-beta
    title "Symbol to Integer Mapping"
    x-axis "Symbol (x)" ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    y-axis "Integer (y)" 0 --> 15
    line "Mapping Values" [0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 8, 9, 10, 11, 6, 8, 1, 12, 13, 1, 14, 4, 15, 0, 0, 4]
```

*Figure 1 – A simplified subset of our non-linear letter-to-index graph.*

---
## Proof-of-Concept Results

We trained a lightweight GRU-based sequence model to automatically resolve ambiguous codes in our custom mapping. The model’s task was to predict the single most likely alphabetic character for each numeric code by evaluating all possible characters, and we manually update the sequence until no brackets remain. For example, the word **“reheat”** is first converted into

```
r[ezv]h[ezv][axy][tqb]
```

The model examines each bracketed code, selects its top candidate (e.g. “e” for `[ezv]`), and we manually update the sequence to

```
reh[ezv][axy][tqb]
```

This prediction-and-replacement cycle continues until the fully decoded word, **“reheat,”** emerges.

The trained model achieved **99.05% accuracy** on a held-out validation set (30% of 800 000 samples, approximately 240 000 phrases).

Additional examples demonstrate the model’s robustness:

1. **Ambiguous input**: `avsqzrdxx`

   * **Iteration 1**: expand to bracketed form

     ```
     [axy][ezv]s[tqb][ezv]rd[axy][axy]
     ```

     model predicts “y” for `[axy]`, yielding

     ```
     y[ezv]s[tqb][ezv]rd[axy][axy]
     ```
   * The cycle continues until the decoded output, **“yesterday,”** is obtained.

2. **Ambiguous input**: `vvvning`

   * **Iteration 1**: expand to bracketed form

     ```
     [ezv][ezv][ezv]ning
     ```

     model predicts “e” for first `[ezv]`, yielding

     ```
     e[ezv][ezv]ning
     ```

   * The cycle continues until the decoded output, **“evening,”** is obtained.

---

## Implications

1. **Mapping as a Cryptographic Key**  
   The non-linear letter-to-index map acts as a secret key. Without knowledge of this mapping, decryption becomes computationally infeasible. Regularly rotating or regenerating the map provides a straightforward mechanism for key refreshment, enhancing security without altering encryption algorithms.

2. **Layered Security Framework**  
   - **Pre-encryption distortion**: Removes linguistic patterns, rendering frequency and n-gram analysis ineffective.  
   - **Standard encryption layer**: Processes already randomized data, compounding security.  
   - **Adversarial requirements**: Attackers must compromise both the mapping scheme and the encryption algorithm to recover plaintext.

---

## Conclusion

This technique is **not** a standalone encryption algorithm. Instead, it introduces a *strictly non-linear symbol-to-numeric mapping* as a secret key, with a trained neural network model serving solely to decode obfuscated plaintext. By integrating AI-driven reconstruction, the method adds a robust defense-in-depth layer when combined with established ciphers like AES or RSA—all while remaining compatible with existing encryption protocols.  

The approach addresses a critical gap in traditional systems: the inherent predictability of standardized mappings. Future work will explore dynamic mapping generation and integration with post-quantum cryptographic schemes.
