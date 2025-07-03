# Advanced Polyphonic Substitution Cipher

## What is This?

This project explores using neural networks to solve an old problem with polyphonic substitution ciphers. In these ciphers, multiple letters can map to the same symbol (like 'e', 'v', and 'z' all becoming '4'), creating ambiguity during decryption. Historically, humans had to figure out which letter was meant from context - a tedious process. This project automates that disambiguation using a GRU neural network.

## Background

Polyphonic substitution ciphers date back to the 16th century. They were interesting because they made frequency analysis harder - if 'e', 'v', and 'z' all encode to '4', you can't just count letter frequencies to crack the code. The downside? Decryption was manual and error-prone, so they weren't practical for regular use.

## How It Works

### The Basic Idea

1. **Encoding**: Map multiple letters to the same number
   - Example: 'e', 'v', 'z' → 4
   - Example: 'a', 'x', 'y' → 0
   
2. **The Problem**: When you see '4', is it 'e', 'v', or 'z'?

3. **The Solution**: Train a neural network to guess based on context

### Simple Example

```
Original: "team"
After encoding: "1 4 0 10" 
Expansion: [tqb][ezv][axy]m
AI predicts: "team" (not "qvxm" or other possibilities)
```

## Implementation Details

### Character Mapping

The system uses these polyphonic groups:
- [e, z, v] → all map to 4
- [a, x, y] → all map to 0  
- [t, q, b] → all map to 1
- [o, j, g] → all map to 6
- [i, k, p] → all map to 8

Other letters have unique mappings.

### The Neural Network

- **Architecture**: GRU with 128 hidden units
- **Training**: 800,000 English phrases
- **Validation**: 240,000 test phrases
- **Accuracy**: 99.05% on test data

### Decryption Process

The AI resolves ambiguities one at a time:

1. Start with ambiguous sequence: `[tqb][ezv][axy]m`
2. Predict first character: 't' from [tqb]
3. Update: `t[ezv][axy]m`
4. Predict next: 'e' from [ezv]
5. Continue until done: `team`

## Results

The system successfully decodes ambiguous text:

- `avsqzrdxx` → `[axy][ezv]s[tqb][ezv]rd[axy][axy]` → `yesterday`
- `vvvning` → `[ezv][ezv][ezv]ning` → `evening`

## Limitations

- Only tested on English text
- Requires the GRU model for decryption (~500MB)
- Processing time increases with message length
- If the AI guesses wrong, the message is corrupted

## Historical Note

This project was inspired by historical polyphonic ciphers used in 16th century Europe and challenges solved by Edgar Allan Poe in the 1840s. The key difference is using AI to automate what was once a manual process.

## Technical Requirements

- Python 3.x
- PyTorch for the GRU model
- ~500MB storage for the trained model
- Basic understanding of substitution ciphers

## Future Ideas

- Test with other languages
- Try different neural architectures
- Experiment with variable-size character groups
- Add error correction for when AI guesses wrong

## Conclusion

This project demonstrates that neural networks can successfully disambiguate polyphonic substitution ciphers, automating what was historically a manual process. While not a complete cryptographic system on its own, it shows how AI can revive old cryptographic techniques that were impractical due to human limitations.

---

**Note**: This is an academic project exploring the intersection of historical cryptography and modern AI. It should not be relied upon as a secure encryption method without further testing and validation.

### References

1. Tomokiyo, S. (2019). "Polyphonic Substitution Cipher." MysteryTwister Challenge.
2. Historical examples from 16th century Italian and French diplomatic correspondence.


---

*This is a student research project. Results are preliminary and would need extensive testing before any practical application.*