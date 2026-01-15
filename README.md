# 📊 **TextFlow** — A 2nd-Order Markov Chain Text Generator

A probabilistic text generation engine built from scratch using **2nd-order Markov Chains** in **pure Python**, demonstrating mathematical foundations of stochastic processes, state transitions, and natural language modeling across diverse literary and technical corpora.

> 🎓 This project was developed as an **educational exploration** of statistical language modeling, Markov processes, and probabilistic text generation, showcasing the mathematical elegance of simple yet powerful algorithms without relying on deep learning frameworks.

---

## 📌 Highlights & Mathematical Foundation

> 🧮 **2nd-Order Markov Property**: The model predicts the next character based on the previous **two characters**, creating a context-aware probability distribution.
>
> 📈 **Sparse Transition Matrices**: Uses dictionary-based sparse representations instead of dense matrices, enabling memory-efficient storage of transition probabilities.
>
> 🎲 **Laplace Smoothing**: Implements additive smoothing (α = 1e-8) to handle unseen character sequences and prevent zero-probability catastrophes.
>
> 🔄 **Atomic File Operations**: YAML state persistence and pickle serialization ensure training resumability and data integrity.
>
> 📚 **Multi-Domain Corpus**: Trained on literary classics (Austen, Shakespeare, Harper Lee), technical texts (ML, robotics), and cultural content (Indian literature, languages).

---

## 🧮 Mathematical Framework

### Markov Chain Theory

A **second-order Markov chain** assumes that the probability of the next state depends only on the two most recent states:

```math
P(X_n \mid X_1, X_2, \ldots, X_{n-1})
=
P(X_n \mid X_{n-2}, X_{n-1})
```

### State Representation

Each state is defined as an ordered pair of consecutive characters:

```math
s_i = (c_{i-1}, c_i)
```

Transitions are defined as probabilities of the next character given the current state:

```math
P(c_{i+1} \mid c_{i-1}, c_i)
```

### Probability Calculations

#### Initial State Distribution

The initial state distribution $\pi$ is computed using additive (Laplace) smoothing:

```math
\pi(s) =
\frac{count(s) + \alpha}
{\sum_{s' \in S} count(s') + |S| \alpha}
```

#### Transition Probabilities

The transition probability from state $s_i$ to state $s_j$ is defined as:

```math
P(s_j \mid s_i) =
\frac{count(s_i \rightarrow s_j) + \alpha}
{\sum_k count(s_i \rightarrow s_k) + |S| \alpha}
```

#### Where

-   $$\alpha$$ — smoothing parameter (default: $$10^{-8}$$)
-   $$S$$ — set of all unique states
-   $$|S|$$ — total number of states
-   $$\text{count}(s_i \rightarrow s_j)$$ — observed transitions from $$s_i$$ to $$s_j$$

### Text Generation Algorithm

1. **Initialization**
   Sample the initial state $s_0$ from the distribution $\pi$.

2. **Propagation**
   For each generation step $t$:

    - Retrieve transition probabilities $P(\cdot \mid s_t)$
    - Sample the next character using weighted random selection

3. **State Update**

```math
s_{t+1} = (c_t, c_{t+1})
```

4. **Termination**
   Stop after generating $n$ characters.

### Anti-Repetition Mechanism

To reduce repetitive loops, a context-aware penalty is applied to recently generated states:

```math
P'(s) = P(s) \cdot \gamma \quad \text{if } s \in recent\_context
```

```math
P'(s) = P(s) \quad \text{otherwise}
```

where:

-   $\gamma \in [0.01, 0.1]$ is the penalty factor
-   `recent_context` denotes a sliding window of recently generated states

This mechanism lowers the probability of repeating recent patterns while preserving overall stochasticity.

---

## 📁 File Structure

```
textflow/
├── main.py                     # 🚀 Main entry point for text generation
├── project.conf                # ⚙️ Project configuration file
├── README.md                   # 📘 Project documentation
├── LICENSE                     # ⚖️  MIT License
├── requirements.txt            # 📦 Python dependencies
├── .gitignore                  # 🚫 Git exclusions
│
├── venv/                       # 🐍 Virtual environment (auto-created)
│
├── utils/                      # 🛠️  Utility modules
│   ├── __init__.py             # Package initialization
│   ├── config.py               # ⚙️  Configuration file parser (.conf)
│   ├── constants.py            # 🔢 Path constants from YAML
│   └── log.py                  # 📝 Logging functionality
│
├── model/                      # 🧠 Training data and model state
│   ├── __init__.py             # Package initialization
│   ├── corpus.txt              # 📚 Training text corpus (~226KB)
│   ├── info.yaml               # ℹ️  Model configuration & metadata
│   ├── params.yaml             # 📊 Training progress tracker
│   ├── trainer.py              # 🏋️  Markov chain trainer
│   │
│   ├── data/                   # 💾 Serialized model weights
│   │   ├── initial.pkl         # Initial state probabilities (π)
│   │   └── transitions.pkl     # Transition probability matrix (P)
│   │
│   └── logs/                   # 📜 Training logs
│       └── sessions.log        # Timestamped training events
│
├── generator/                  # 🎲 Text generation engine
│   ├── __init__.py             # Package initialization
│   ├── markov.py               # 🔮 Core Markov generator class
│   └── debugger.py             # 🐛 Model diagnostics & debugging
│
└── samples/                    # 🖼️  Demo outputs & screenshots
    └── (sample images)         # Generated text examples
```

---

## ⚙️ Core Components

### 1. **Trainer (`model/trainer.py`)**

-   **Incremental Training**: Resumes from last checkpoint using cursor-based file streaming
-   **Chunked Processing**: Reads corpus in 64KB chunks to handle large files
-   **Word-Level Bigrams**: Processes text as word pairs `(wᵢ₋₁, wᵢ) → wᵢ₊₁`
-   **Atomic State Persistence**: YAML + Pickle for crash-resistant training
-   **Progress Tracking**: tqdm integration for visual feedback

### 2. **Generator (`generator/markov.py`)**

-   **Dictionary-Based Transitions**: Sparse storage (only non-zero probabilities)
-   **Memory Optimization**: Limits to top 8000 most active states
-   **Multiple Sampling Modes**:
    -   Deterministic (argmax selection)
    -   Stochastic (weighted random sampling)
    -   Context-aware (anti-repetition penalties)
-   **Smoothing**: Laplace smoothing prevents zero-probability failures

### 3. **Configuration System**

-   **`info.yaml`**: Model hyperparameters (order=2, preprocessing flags)
-   **`params.yaml`**: Training state (cursor position, word count, line number)
-   **Atomic Updates**: Temp file + `os.replace()` for crash safety

### 4. **Logging (`utils/log.py`)**

-   **Dual Output**: Writes to both file and console
-   **YAML-Configured**: Log level, format, and file path from `info.yaml`
-   **Structured Messages**: DEBUG, INFO, WARNING, ERROR, CRITICAL levels

---

## 🧪 Installation & Setup

### Prerequisites

-   Python 3.8+ (tested on 3.10)
-   pip package manager
-   Git (for cloning)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/textflow.git
cd textflow

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the generator
python main.py
```

---

## 📦 `requirements.txt`

```txt
# Core dependencies
numpy>=1.24.0,<2.0.0
pyyaml>=6.0
tqdm>=4.65.0

# Optional (for debugging)
matplotlib>=3.7.0  # Visualization of probability distributions
pandas>=2.0.0      # Data analysis utilities

# Development dependencies (optional)
pytest>=7.3.0      # Testing framework
black>=23.0.0      # Code formatting
```

---

## 🚀 Usage Examples

### Basic Text Generation

```python
from generator import Markov

# Initialize generator (trains if needed)
markov = Markov(max_states=8000, smoothing_alpha=1e-8)

# Generate 50 words
text = markov.generate_words(num_words=50, deterministic=False)
print(text)
```

### Advanced Options

```python
# Deterministic generation (always picks highest probability)
text_det = markov.generate_words(num_words=30, deterministic=True)

# With anti-repetition (reduces loops)
text_varied = markov.generate_words(
    num_words=100,
    anti_repetition=True,
    context_window=5
)

# Get model statistics
stats = markov.get_vocabulary_stats()
print(f"States: {stats['num_states']}, Vocab: {stats['vocabulary_size']}")
```

### Debugging & Analysis

```bash
# Run comprehensive model diagnostics
python -m generator.debugger
```

This will output:

-   Model statistics (states, vocabulary, transitions)
-   Top initial bigrams
-   Dead-end states analysis
-   Sample generations with different methods
-   Improvement suggestions

---

## 🧠 Model Architecture

### State Space Design

**State Definition:**

```python
State = Tuple[str, str]  # (word_i-1, word_i)
```

**Example States:**

```
("the", "quick") → ["brown", "lazy", "slow"]
("quick", "brown") → ["fox"]
```

### Probability Storage

**Initial Distribution (π):**

```python
{
    ("it", "is"): 0.0342,
    ("the", "world"): 0.0156,
    ...
}
```

**Transition Matrix (P):**

```python
{
    ("it", "is"): {
        ("is", "a"): 0.45,
        ("is", "not"): 0.23,
        ("is", "the"): 0.18,
        ...
    },
    ...
}
```

### Training Process

```
┌─────────────┐
│ Read Corpus │ (streaming, 64KB chunks)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Tokenize Words  │ (lowercase, alphabetic only)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Count Bigrams    │ (prev_prev, prev) → current
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Update Pickles   │ (atomic write with .tmp)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Save Checkpoint  │ (cursor, line, count in YAML)
└──────────────────┘
```

---

## 🔬 Experimental Features

### 1. **Adaptive Smoothing**

Dynamically adjusts α based on corpus size:

```python
α = max(1e-8, 1.0 / sqrt(vocab_size))
```

### 2. **Context Tracking**

Maintains sliding window of recent states to avoid repetition loops.

### 3. **State Pruning**

Limits model to top-N most frequent states (default: 8000) to prevent memory explosion.

### 4. **Probability Validation**

Ensures all transition rows sum to 1.0 (within numerical tolerance).

---

## 📊 Performance Metrics

| Metric           | Value                      |
| ---------------- | -------------------------- |
| Corpus Size      | ~226 KB (raw text)         |
| Vocabulary       | ~35,000 unique words       |
| States           | ~8,000 (after pruning)     |
| Transitions      | ~150,000 (sparse)          |
| Training Time    | ~2-5 seconds (incremental) |
| Generation Speed | ~1000 words/sec            |
| Memory Usage     | ~50 MB (loaded model)      |

---

## 🎨 Sample Outputs

<p align="center">
  <img src="sample/001.jpg" alt="text-flow output demo" width="800"/>
</p>

---

## 🐛 Debugging Tools

### `debugger.py` Output Structure

```
============================================================
MODEL STATISTICS
============================================================
num_states               : 8000
vocabulary_size          : 34775
initial_states           : 7845
...

============================================================
RAW DATA ANALYSIS
============================================================
Initial bigram states: 7845
Top 10 initial bigrams:
  1. (it          , is          ) ->  342 times
  2. (the         , world       ) ->  156 times
  ...

============================================================
IMPROVEMENT SUGGESTIONS
============================================================
⚠️  Very sparse transitions - model may produce repetitive text
   Try: Increase smoothing_alpha or add more training data
✅  Contains 'Pride and Prejudice' vocabulary
```

---

## 🔧 Configuration Files

### `info.yaml`

```yaml
model:
    type: "MarkovChain"
    level: "character" # or "word"
    order: 2
    preprocess:
        alphabetic_only: true
        lowercase: true
        remove_punctuation: true
        space_normalization: true

paths:
    info: "model/info.yaml"
    corpus: "model/corpus.txt"
    params: "model/params.yaml"
    initial: "model/data/initial.pkl"
    transitions: "model/data/transitions.pkl"
    logs: "model/logs/sessions.log"

logging:
    level: INFO
    format: "%(asctime)s [%(levelname)s] %(message)s"
```

### `params.yaml` (Auto-Updated)

```yaml
corpus:
    cursor: 225801 # Byte position in corpus
    count: 34775 # Total words processed
    line: 2777 # Lines processed
```

---

## 🚧 Known Limitations

1. **Repetition Loops**: Despite anti-repetition measures, the model can occasionally fall into cycles (inherent to low-order Markov chains)
2. **Grammar Awareness**: No syntactic understanding—output may be grammatically incorrect
3. **Context Window**: Limited to 2 characters (increasing order exponentially increases memory)
4. **Sparse Data**: Rare character combinations may generate unlikely sequences
5. **No Semantic Understanding**: Purely statistical; doesn't "understand" meaning

---

## 🔮 Future Enhancements

-   [ ] **Higher-Order Models**: Experiment with 3rd/4th-order Markov chains
-   [ ] **Hybrid Approach**: Combine character and word-level models
-   [ ] **Temperature Sampling**: Add temperature parameter for creativity control
-   [ ] **Beam Search**: Implement beam search for more coherent generation
-   [ ] **Corpus Expansion**: Add more diverse training data
-   [ ] **Interactive Mode**: Live web interface for text generation
-   [ ] **Model Comparison**: Benchmark against GPT-2/LSTM baselines

---

## 📚 Educational Value

This project demonstrates:

✅ **Stochastic Processes**: Practical application of Markov chains  
✅ **Probability Theory**: Conditional probabilities, smoothing, normalization  
✅ **Algorithm Design**: State space management, sparse storage, incremental training  
✅ **Software Engineering**: Modular design, logging, configuration management, atomic operations  
✅ **Data Structures**: Hash maps, sparse matrices, probability distributions

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

-   Additional corpus sources (more languages, domains)
-   Alternative smoothing techniques (Kneser-Ney, Good-Turing)
-   Visualization tools (state transition graphs, probability heatmaps)
-   Performance optimizations (Cython, multiprocessing)
-   Unit tests and benchmarks

---

## ⚖️ License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute this software with proper attribution.

---

## 👨‍💻 Author

> **Anvay Mayekar**  
> 🎓 B.Tech in Electronics & Computer Science — SAKEC, Mumbai
>
> [![GitHub](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://www.github.com/anvaymayekar) > [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://in.linkedin.com/in/anvaymayekar) > [![Gmail](https://img.shields.io/badge/Gmail-D14836.svg?style=for-the-badge&logo=gmail&logoColor=white)](mailto:anvaay@gmail.com)
