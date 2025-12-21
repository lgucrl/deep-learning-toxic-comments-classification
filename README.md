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

1. **Exploratory Data Analysis (EDA)**  
   The workflow starts by inspecting dataset shape, column types, and missing values. Since this is a **multi-label classification**, the analysis focuses on label frequencies, how many labels appear per single comment and how often labels co-occur (e.g., correlations between toxic, obscene, and insult). The EDA also examines comment **length statistics** to guide later modeling choices (like sequence length), and it highlights the strong imbalance between "clean" vs. "toxic" samples, an important factor that affects precision/recall behavior during evaluation.

2. **Text cleaning and normalization**  
   Raw comments are cleaned with a preprocessing pipeline that removes digits and control characters and drops English stopwords using NLTK. This step reduces noise and average comment length and helps the model focus on informative tokens. The cleaned text is saved in a separate column so the same transformation can be reused consistently for training and evaluation.

3. **Multi-label stratified train/test split**  
   Rather than a standard random split, the project uses **MultilabelStratifiedShuffleSplit** to create train/test sets (with a 80/20 ratio) while maintaining similar multi-label distributions across splits. This is crucial in imbalanced, multi-label settings: without stratification, rare labels can become underrepresented in the test set, making evaluation unreliable.

4. **Vectorization into fixed-length integer sequences**  
   The cleaned comments are converted into sequences of token IDs using `tf.keras.layers.TextVectorization`. The vocabulary size is capped to cover **90%** of word occurrences, and the output sequence length is set to the **95th percentile** of comment length to increase efficiency without excessive truncation. This results in uniform tensors (`(N, 230)` sequences) suitable for recurrent layers.

5. **Handling class imbalance with multi-label oversampling**  
   Because toxicity labels are highly imbalanced, the training set is balanced via **MLSMOTE** (Multi-Label SMOTE). :contentReference[oaicite:14]{index=14} The approach generates synthetic samples for minority/toxic label combinations and then merges them with a sampled subset of clean comments to target a ~50/50 clean-to-toxic training distribution, while explicitly keeping the **test set unmodified** for realistic evaluation. :contentReference[oaicite:15]{index=15}

   To mitigate the dominance of clean comments and improve learning on minority labels, the training set is balanced with MLSMOTE (Multi-Label SMOTE). Synthetic samples are generated for minority/toxic combinations and then combined with a sampled subset of clean comments to target a ~50/50 clean-to-toxic training distribution. Because the synthetic process can produce non-integer token values, generated sequences are rounded and clipped to valid token ID ranges before modeling.

   ---

6. **Model design and training (RNN with Bidirectional GRUs)**  
   The network uses an **Embedding** layer (with masking for padding token 0), **SpatialDropout1D** regularization, and stacked **Bidirectional GRU** layers to capture context from both directions in the sequence. :contentReference[oaicite:16]{index=16} A dense head ends with a **sigmoid** layer producing six independent probabilities (one per label). :contentReference[oaicite:17]{index=17} The model is trained with **binary cross-entropy** and optimized with **Adam**, tracking metrics such as accuracy, precision, recall, and weighted F1, with **early stopping** on validation loss. :contentReference[oaicite:18]{index=18}

7. **Evaluation on test set and prediction examples**  
   After training, evaluation is performed on the **original (unbalanced) test set** by computing probability outputs, converting them to binary predictions with a **0.5 threshold**, and reporting overall (weighted) metrics plus per-label metrics. :contentReference[oaicite:19]{index=19} Performance is further analyzed with **confusion matrices** and threshold-sensitive curves such as **ROC** and **Precision–Recall** (especially informative under heavy imbalance). :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}  
   To make the model easier to “try out,” the project also implements a small inference utility that takes an arbitrary text comment, applies the same **cleaning + vectorization** pipeline, runs the model to obtain probabilities and binary decisions, and visualizes label probabilities in a bar chart with a horizontal threshold line. :contentReference[oaicite:22]{index=22}

---

## Tech stack
