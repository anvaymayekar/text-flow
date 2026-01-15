from utils import log
from generator import Markov
import pickle
from pathlib import Path


def debug_markov_model():
    """Comprehensive debugging of the Markov model."""

    log("Starting Markov model debugging...", mode="INFO")

    try:
        markov = Markov()

        print("\n" + "=" * 60)
        print("MODEL STATISTICS")
        print("=" * 60)

        stats = markov.get_vocabulary_stats()
        for key, value in stats.items():
            print(f"{key:25}: {value}")

        # Debug the raw data
        print("\n" + "=" * 60)
        print("RAW DATA ANALYSIS")
        print("=" * 60)

        # Load and inspect initial counts
        if Path(markov.initial_file).exists():
            with open(markov.initial_file, "rb") as f:
                initial_counts = pickle.load(f)

            print(f"\nInitial bigram states: {len(initial_counts)}")
            print("Top 10 initial bigrams:")
            sorted_initial = sorted(
                initial_counts.items(), key=lambda x: x[1], reverse=True
            )
            for i, ((w1, w2), count) in enumerate(sorted_initial[:10]):
                print(f"  {i+1:2d}. ({w1:<12}, {w2:<12}) -> {count:4d} times")

        # Load and inspect transition counts
        if Path(markov.transitions_file).exists():
            with open(markov.transitions_file, "rb") as f:
                transition_counts = pickle.load(f)

            print(f"\nTransition patterns: {len(transition_counts)}")
            print("States with most transitions:")

            # Find states with most outgoing transitions
            state_transition_counts = [
                (state, len(transitions))
                for state, transitions in transition_counts.items()
            ]
            state_transition_counts.sort(key=lambda x: x[1], reverse=True)

            for i, (state, count) in enumerate(state_transition_counts[:5]):
                print(f"  {i+1}. {state} -> {count} different next states")

            print("\nStates with only 1 transition (potential dead ends):")
            dead_ends = [
                (state, list(transitions.keys())[0])
                for state, transitions in transition_counts.items()
                if len(transitions) == 1
            ]
            print(f"  Found {len(dead_ends)} dead-end states")
            for i, (from_state, to_state) in enumerate(dead_ends[:5]):
                print(f"    {from_state} -> {to_state}")

        print("\n" + "=" * 60)
        print("GENERATION ANALYSIS")
        print("=" * 60)

        # Test different generation methods
        methods = [
            (
                "Simple Deterministic",
                lambda: markov.generate_words_simple(15, deterministic=True),
            ),
            (
                "Simple Random",
                lambda: markov.generate_words_simple(15, deterministic=False),
            ),
            (
                "State Propagation Det",
                lambda: markov.generate_words(
                    15, deterministic=True, use_state_propagation=True
                ),
            ),
            (
                "State Propagation Rand",
                lambda: markov.generate_words(
                    15, deterministic=False, use_state_propagation=True
                ),
            ),
            (
                "Context Tracking",
                lambda: markov.generate_with_context_tracking(15, context_window=3),
            ),
        ]

        for method_name, method_func in methods:
            try:
                text = method_func()
                print(f"\n{method_name:20}: {text}")
            except Exception as e:
                print(f"\n{method_name:20}: ERROR - {e}")

        # Analyze vocabulary distribution
        print("\n" + "=" * 60)
        print("VOCABULARY ANALYSIS")
        print("=" * 60)

        # Count word frequencies in initial states
        word_freq = {}
        for (w1, w2), count in initial_counts.items():
            word_freq[w1] = word_freq.get(w1, 0) + count
            word_freq[w2] = word_freq.get(w2, 0) + count

        print("Most frequent words in model:")
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (word, freq) in enumerate(sorted_words[:15]):
            print(f"  {i+1:2d}. {word:<15} -> {freq:4d} times")

        # Check for common problematic words
        problematic = ["and", "the", "of", "to", "a", "in", "is", "it", "that", "for"]
        print(f"\nStop words frequency:")
        for word in problematic:
            if word in word_freq:
                print(f"  {word:<10}: {word_freq[word]:4d} times")

        return markov

    except Exception as e:
        log(f"Debug failed: {e}", mode="ERROR")
        return None


def suggest_improvements(markov):
    """Analyze the model and suggest improvements."""

    print("\n" + "=" * 60)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    try:
        stats = markov.get_vocabulary_stats()

        # Check data quality indicators
        vocab_size = stats["vocabulary_size"]
        num_states = stats["num_states"]
        non_zero_transitions = stats["non_zero_transitions"]

        print(f"Model Analysis:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Bigram states: {num_states}")
        print(f"  Non-zero transitions: {non_zero_transitions}")
        print(
            f"  Sparsity: {(non_zero_transitions / (num_states * num_states) * 100):.2f}% filled"
        )

        print(f"\nRecommendations:")

        if vocab_size < 1000:
            print("  ⚠️  Small vocabulary - consider adding more diverse text")

        if num_states < vocab_size:
            print("  ⚠️  Very few bigram states - text may be too fragmented")

        sparsity = non_zero_transitions / (num_states * num_states)
        if sparsity < 0.01:
            print("  ⚠️  Very sparse transitions - model may produce repetitive text")
            print("     Try: Increase smoothing_alpha or add more training data")

        if "pride" in str(markov._vocab) and "prejudice" in str(markov._vocab):
            print("  ✅  Contains 'Pride and Prejudice' vocabulary")
            print("     The repetitive output suggests insufficient context diversity")

        print(f"\n  Suggested fixes:")
        print(f"  1. Clean your corpus - remove metadata, headers, chapter markers")
        print(f"  2. Ensure continuous flowing text rather than snippets")
        print(f"  3. Try higher smoothing: Markov(smoothing_alpha=1e-6)")
        print(f"  4. Use context tracking to reduce repetition")
        print(f"  5. Consider preprocessing to remove very common words")

    except Exception as e:
        log(f"Analysis failed: {e}", mode="ERROR")


def test_corpus_quality():
    """Test the quality of the training corpus."""

    print("\n" + "=" * 60)
    print("CORPUS QUALITY CHECK")
    print("=" * 60)

    try:
        markov = Markov()

        # Sample some text to see what we're working with
        print("Testing generation with different parameters...")

        # Try with higher smoothing
        print("\nTrying with higher smoothing:")
        markov_smooth = Markov(smoothing_alpha=1e-5)
        text_smooth = markov_smooth.generate_words_simple(20, deterministic=False)
        print(f"Higher smoothing: {text_smooth}")

        # Try context tracking
        print("\nTrying context tracking:")
        text_context = markov.generate_with_context_tracking(20, context_window=5)
        print(f"Context tracking: {text_context}")

    except Exception as e:
        log(f"Corpus quality test failed: {e}", mode="ERROR")


if __name__ == "__main__":
    # Run comprehensive debugging
    markov = debug_markov_model()

    if markov:
        suggest_improvements(markov)
        test_corpus_quality()
