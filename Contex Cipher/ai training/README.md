# Context Cipher: Training Overview

## Model Architecture
- **Type**: Sequence-to-sequence bidirectional GRU  
- **Layers**: 3  
- **Mechanism**: Attention-enhanced decoding  

## Training Data Format
CSV columns: `input`, `target`, `original`  
- **Input**: Cluster tokens with positional ambiguity (e.g., `[tqb][ezv]`)  
- **Target**: Next character to resolve ambiguity  
- **Original**: Full plaintext reference  

### Example Pre-Train Samples
```
be[ojg][ojg][ezv]d,g,begged  
beg[ojg][ezv]d,g,begged  
begg[ezv]d,e,begged
```

### Example Post-Train Samples
```
[tqb][ezv][ezv]n [axy][tqb]l[ezv],b,been able  
[tqb][ojg] r[ezv]s[ikp]s[tqb],t,to resist  
[tqb]u[axy][ikp]n[ojg] [axy] f[ezv],b,buying a few
```

## Training Phases
1. **Pre-Training**  
   - **Samples**: 200k word-level sequences  
   - **Focus**: Basic cluster-to-character resolution  
   - **Validation Accuracy**: 94% 

2. **Fine-Tuning**  
   - **Samples**: 800k phrase-level sequences  
   - **Focus**: Context-aware disambiguation  
   - **Final Accuracy**: 99.05%  

## Datasets
- Primary: BookCorpus (phrases/sentences)  
- Supplemental: `words_alpha.txt` (word-level patterns)  

---

*— End of Training Overview —*  
