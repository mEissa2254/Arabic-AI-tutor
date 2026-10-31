# Arabic AI Tutor: Fine-Tuned Transformer Models for Question Answering and Text Summarization

An Arabic NLP tutoring system that combines **multi-domain abstractive text summarization** and **generative question answering**.

# Arabic AI Tutor: Fine-Tuned Transformer Models for Question Answering and Text Summarization

An Arabic Natural Language Processing (NLP) tutoring system that combines **multi-domain abstractive text summarization** and **generative question answering**. This research investigates how domain-aware conditioning improves Arabic summarization and evaluates parameter-efficient fine-tuning for generative QA on ArabicMMLU.

> **Research Project** — Department of Computer Science, Nile University, Cairo, Egypt.

---

## Overview

Arabic NLP remains less developed than English NLP, particularly for educational applications. This project explores an AI Arabic Tutor with two complementary capabilities:

1. **Arabic Text Summarization:** Produces abstractive summaries of Arabic articles from multiple content domains.
2. **Arabic Question Answering:** Generates free-text answers to educational questions across five ArabicMMLU subjects.

The system is designed as a multi-agent Streamlit application. A domain router directs the user's input to the summarization or question-answering pipeline. Both models are intended to run locally, without requiring paid inference APIs.

## Main Contributions

- A controlled comparison of **AraBART** and **AraT5v2** for Arabic abstractive summarization, with and without domain-prefix conditioning.
- A TF-IDF + Logistic Regression domain classifier that predicts an article's domain and supplies it to the summarization model as a T5-style prefix.
- Generative sequence-to-sequence fine-tuning on five ArabicMMLU subjects using **LoRA**.
- An evaluation of **AraT5v2-base + LoRA** and **mT5-base + LoRA** for generative Arabic question answering.
- Per-domain summarization analysis and qualitative error analysis for both tasks.

---

## System Architecture

```text
                         Arabic User Input
                                |
                          Domain / Task Router
                         /                    \
                        /                      \
             Summarization Agent           QA Agent
                     |                         |
             Domain Classifier            Prompt Formatting
          (TF-IDF + Logistic Regression)        |
                     |                    mT5-base + LoRA
             Domain Prefix Injection             |
                     |                     Generated Answer
              AraT5v2 Summarizer
                     |
              Generated Summary
```

### 1. Summarization Pipeline

The summarization pipeline compares two encoder-decoder models:

- **AraBART**
- **AraT5v2-base**

Each model is evaluated in two configurations:

**Standard:**

```text
summarize: <article>
```

**Domain-conditioned:**

```text
summarize [domain]: <article>
```

The domain classifier uses character-level TF-IDF features (character n-grams 1–2, up to 45,000 features) and Logistic Regression.

Reported classifier performance:

- **Test Accuracy:** 95.59%
- **F1-score:** 95.57%

For generation, the experiments use beam search with 4 beams and a no-repeat n-gram size of 3.

### 2. Question-Answering Pipeline

The QA task uses a generative sequence-to-sequence formulation. Each question is formatted using the prefix:

```text
question: [QUESTION] answer:
```

The target is the text of the correct answer option.

LoRA is applied to the attention and feed-forward projection matrices:

- `q`
- `k`
- `v`
- `o`
- `wi`
- `wo`

The evaluated models are:

- AraT5v2-base
- mT5-base

Reported LoRA configuration:

| Parameter | Value |
|---|---:|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |

---

## Datasets

### ArabSummaries

The summarization experiments use the ArabSummaries dataset.

| Item | Count |
|---|---:|
| Original samples | 7,000 |
| Samples after filtering | 6,118 |
| Training | 4,894 |
| Validation | 612 |
| Test | 612 |

The paper describes the original dataset as containing seven balanced domains, with 1,000 samples per domain:

- Culture
- Finance
- Medical
- Politics
- Religion
- Sports
- Technology

The split is 80/10/10 and stratified by domain.

### ArabicMMLU

The QA experiments use five subjects from ArabicMMLU.

| Item | Count |
|---|---:|
| Original questions | 28,910 |
| Unique questions after deduplication | 14,123 |
| Selected training samples | 6,129 |
| Validation samples | 679 |

Selected subjects:

- Islamic Studies
- Biology
- Geography
- History
- Social Science

The paper reports 14,787 duplicates removed before selecting the five subjects.

---

## Experimental Setup

### Summarization

| Configuration | Value |
|---|---|
| Framework | Hugging Face Seq2SeqTrainer |
| Maximum input length | 512 tokens |
| Maximum target length | 128 tokens |
| Mixed precision | FP16 |
| GPU environment | Kaggle Tesla T4 (15.6 GB VRAM) |
| Early stopping | ROUGE-L, patience 3 |
| AraBART learning rate | 5e-5 |
| AraT5v2 learning rate | 3e-4 |

### Question Answering

| Configuration | Value |
|---|---|
| Fine-tuning method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Warmup ratio | 0.12 |
| Maximum epochs | 8 |
| mT5 precision | BF16 |
| AraT5v2 precision | FP16 |

The paper reports **4,620,288 trainable parameters** for the LoRA setup.

---

## Results

### Arabic Abstractive Summarization

Results on the ArabSummaries test set (`n = 612`):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| AraBART — Standard | 46.58 | 26.58 | 39.19 |
| AraBART + Domain | 48.82 | 29.08 | 41.60 |
| AraT5v2 — Standard | 49.92 | 30.45 | 43.00 |
| **AraT5v2 + Domain** | **51.05** | **31.33** | **43.86** |

**Best summarization configuration:** AraT5v2 with domain conditioning.

Domain conditioning improved ROUGE-1 by:

- **+2.24 points** for AraBART (46.58 → 48.82).
- **+1.13 points** for AraT5v2 (49.92 → 51.05).

### Generative Question Answering

Results on the ArabicMMLU validation set (`n = 679`):

| Model | Validation Loss | ROUGE-1 | ROUGE-L | BLEU |
|---|---:|---:|---:|---:|
| AraT5v2 + LoRA | 2.303 | — | — | — |
| **mT5-base + LoRA** | **0.089** | **64.01** | **64.01** | **72.77** |

mT5-base + LoRA converged successfully, while AraT5v2 + LoRA did not converge in the reported QA experiment.

### Key Findings

- Domain-prefix conditioning consistently improved summarization scores for both tested architectures.
- AraT5v2 + domain conditioning achieved the strongest summarization result in the experiments.
- mT5-base + LoRA was the successful model configuration for generative QA.
- The results indicate that model/task compatibility matters: the strongest summarization model was not the strongest QA model.

---

## Error Analysis and Limitations

The paper identifies several limitations that are important when interpreting the results.

### Summarization

In the analysis of 200 AraBART test predictions:

| Error Type | Percentage |
|---|---:|
| Length violations | 70% |
| Information omission | 43.5% |
| Hallucination | 35% |
| Redundancy | 1.5% |
| Semantic errors | 3% |

Long input articles were more affected by information omission. The paper links this issue to the 512-token input truncation limit.

### Question Answering

In the analysis of 679 mT5 validation predictions:

| Error Type | Percentage |
|---|---:|
| Answer paraphrasing | 31.2% |
| Prefix leakage | 18.4% |
| Over-generation | 14.1% |
| Incorrect factual generation | 11.6% |

ROUGE and BLEU rely on lexical overlap, so they can penalize semantically correct answers expressed with different Arabic wording. The system should therefore not be treated as a source of guaranteed factual answers.

### Scope and Responsible Use

- The reported results are benchmark and research results, not evidence of performance in every real-world educational setting.
- Generated summaries and answers should be reviewed for factual accuracy.
- The paper identifies long-context truncation and subject-specific QA improvement as future work.
- The paper does not report a public hosted demo URL or a complete software dependency list.

---

## Technologies and Methods

- Python
- Natural Language Processing (NLP)
- Hugging Face Transformers
- AraBART
- AraT5v2
- mT5
- LoRA (Parameter-Efficient Fine-Tuning)
- TF-IDF
- Logistic Regression
- Streamlit
- ROUGE
- BLEU

---

## Future Work

Potential directions identified by the research include:

- Improving summarization performance on long Arabic articles.
- Reducing information omission caused by input truncation.
- Improving QA performance across individual subject domains.
- Further investigating model and task compatibility.

---


---

## Acknowledgements

The authors acknowledge the Kaggle platform for providing access to free GPU resources, including the Tesla T4 GPU, and the creators of the datasets and pretrained models used in this research.

---

## Citation

If you use or refer to this research, please cite the paper:

```text
Arabic AI Tutor: Fine-Tuned Transformer Models for Question Answering and Text Summarization.
