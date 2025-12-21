# Deep Learning multi-label classification of online toxic comments

This deep learning project builds a detection and classification system for **toxic comments** on social platforms: given an online comment, the model predicts whether it contains one or more types of toxicity. The classifier is implemented in **Python** using **TensorFlow/Keras** and outputs a **6-dimensional vector of binary labels** (one per toxicity category). 

---

## Dataset

The project uses a dataset of **~160,000 English comments** with **multi-label annotations** for toxicity. Each comment can belong to **zero, one, or multiple** categories among:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

In addition to the label columns, an aggregated field (`sum_injurious`) is used to summarize how many toxicity labels are present for each comment. The dataset is **strongly imbalanced**: about **90%** of comments are “clean” (no toxicity labels), and only ~10% contain at least one toxic label, with some labels (notably `threat`) being particularly rare.

---

## Project workflow
