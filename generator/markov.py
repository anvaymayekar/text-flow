from __future__ import annotations
import numpy as np
from model import Trainer, TrainerError
from utils import log
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm


class GeneratorError(Exception):
    """Raised for errors during word-level bigram generation."""

    pass


class Markov(Trainer):
    """Memory-efficient 2nd-order Markov word generator"""

    def __init__(
        self,
        chunk_bytes: int = 64 * 1024,
        use_tqdm: bool = True,
        smoothing_alpha: float = 1e-8,
        max_states: int = 8000,
    ) -> None:
        super().__init__(chunk_bytes=chunk_bytes, use_tqdm=use_tqdm)
        self.smoothing_alpha = smoothing_alpha
        self.max_states = max_states  # Limit states to prevent memory issues

        log("Initializing memory-efficient 2nd-order WordMarkovGenerator...", "INFO")

        try:
            self.run_training()  # This calls your existing trainer
        except Exception as e:
            log(f"Training failed during generator initialization: {e}", "ERROR")
            raise TrainerError("Failed to initialize WordMarkovGenerator")

        # Load with dictionary-based approach (no dense matrices)
        self._initial_probs, self._transitions, self._vocab, self._all_states = (
            self._load_probabilities_as_dicts()
        )

    def _load_probabilities_as_dicts(
        self,
    ) -> Tuple[
        Dict[Tuple[str, str], float],
        Dict[Tuple[str, str], Dict[Tuple[str, str], float]],
        set,
        List[Tuple[str, str]],
    ]:
        """Load probabilities as dictionaries instead of matrices to save memory."""

        # Load initial counts from your existing pickle files
        if Path(self.initial_file).exists():
            with open(self.initial_file, "rb") as f:
                initial_counts: Dict[Tuple[str, str], int] = pickle.load(f)
        else:
            initial_counts = {}

        # Load transition counts from your existing pickle files
        if Path(self.transitions_file).exists():
            with open(self.transitions_file, "rb") as f:
                transition_counts: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = (
                    pickle.load(f)
                )
        else:
            transition_counts = {}

        log(
            f"Raw data: {len(initial_counts)} initial, {len(transition_counts)} transitions",
            "INFO",
        )

        # Handle the memory problem by filtering to most active states
        if len(transition_counts) > self.max_states:
            log(
                f"Too many states ({len(transition_counts)}), filtering to {self.max_states}",
                "WARNING",
            )

            # Calculate state activity (total transitions)
            state_activity = {}
            for from_state, to_dict in transition_counts.items():
                state_activity[from_state] = sum(to_dict.values())

            # Keep most active states
            top_states = sorted(
                state_activity.items(), key=lambda x: x[1], reverse=True
            )
            keep_states = set(state[0] for state in top_states[: self.max_states])

            # Filter transitions to keep only active states
            filtered_transitions = {}
            for from_state, to_dict in transition_counts.items():
                if from_state in keep_states:
                    filtered_to_dict = {}
                    for to_state, count in to_dict.items():
                        if (
                            to_state in keep_states
                        ):  # Keep only if both states are active
                            filtered_to_dict[to_state] = count
                    if filtered_to_dict:  # Only add if there are valid transitions
                        filtered_transitions[from_state] = filtered_to_dict

            transition_counts = filtered_transitions

            # Also filter initial counts
            initial_counts = {
                state: count
                for state, count in initial_counts.items()
                if state in keep_states
            }

            log(f"Filtered to {len(transition_counts)} transition states", "INFO")

        # Build vocabulary and state list
        vocab = set()
        all_states = set()

        for state in initial_counts.keys():
            all_states.add(state)
            vocab.add(state[0])
            vocab.add(state[1])

        for from_state, to_dict in transition_counts.items():
            all_states.add(from_state)
            vocab.add(from_state[0])
            vocab.add(from_state[1])
            for to_state in to_dict.keys():
                all_states.add(to_state)
                vocab.add(to_state[0])
                vocab.add(to_state[1])

        all_states_list = sorted(list(all_states))
        log(f"Final model: {len(all_states)} states, {len(vocab)} vocabulary", "INFO")

        # Convert counts to probabilities (dictionary-based, not matrices)

        # Initial probabilities
        initial_probs = {}
        total_initial = sum(initial_counts.values()) if initial_counts else 1

        for state in all_states_list:
            count = initial_counts.get(state, 0)
            # Add smoothing
            prob = (count + self.smoothing_alpha) / (
                total_initial + len(all_states) * self.smoothing_alpha
            )
            initial_probs[state] = prob

        # Transition probabilities
        transitions = {}
        for from_state in all_states_list:
            to_dict = transition_counts.get(from_state, {})
            row_total = sum(to_dict.values()) if to_dict else 0

            row_probs = {}
            for to_state in all_states_list:
                count = to_dict.get(to_state, 0)
                # Add smoothing
                prob = (count + self.smoothing_alpha) / (
                    row_total + len(all_states) * self.smoothing_alpha
                )
                if prob > 1e-10:  # Only store non-tiny probabilities
                    row_probs[to_state] = prob

            if row_probs:
                transitions[from_state] = row_probs

        return initial_probs, transitions, vocab, all_states_list

    def _sample_from_dict(
        self, prob_dict: Dict[Tuple[str, str], float], deterministic: bool = False
    ) -> Tuple[str, str]:
        """Sample from a probability dictionary."""
        if not prob_dict:
            raise GeneratorError("Cannot sample from empty probability dict")

        states = list(prob_dict.keys())
        probs = list(prob_dict.values())

        # Normalize
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / len(probs)] * len(probs)
        else:
            probs = [p / total for p in probs]

        if deterministic:
            max_idx = probs.index(max(probs))
            return states[max_idx]
        else:
            # Manual sampling to avoid numpy.choice issues with tuples
            cumsum = 0
            rand_val = np.random.random()
            for i, prob in enumerate(probs):
                cumsum += prob
                if rand_val <= cumsum:
                    return states[i]
            # Fallback (shouldn't happen with proper normalization)
            return states[-1]

    def _find_valid_next_states(
        self, current_word: str
    ) -> Dict[Tuple[str, str], float]:
        """Find states that can follow the current word (for 2nd-order consistency)."""
        valid_states = {}

        for state, prob in self._initial_probs.items():
            if state[0] == current_word:  # State starts with our current word
                valid_states[state] = prob

        return valid_states

    def generate_words(
        self,
        num_words: int,
        deterministic: bool = False,
        anti_repetition: bool = True,
        context_window: int = 4,
    ) -> str:
        """
        Generate words using dictionary-based probabilities (memory efficient).
        """
        if num_words < 1:
            raise GeneratorError("num_words must be >= 1")

        if not self._initial_probs:
            raise GeneratorError("No initial states available")

        words_generated = []
        recent_words = []  # Track recent words to avoid repetition

        # Sample initial state
        current_state = self._sample_from_dict(self._initial_probs, deterministic)
        words_generated.extend([current_state[0], current_state[1]])
        recent_words.extend([current_state[0], current_state[1]])

        if num_words <= 2:
            return " ".join(words_generated[:num_words])

        # Generate remaining words
        for step in tqdm(
            range(2, num_words), desc="Generating", disable=not self.use_tqdm
        ):

            # Get transition probabilities
            if current_state in self._transitions:
                transition_probs = self._transitions[current_state].copy()
            else:
                # Use valid next states as fallback
                valid_states = self._find_valid_next_states(current_state[1])
                transition_probs = (
                    valid_states if valid_states else self._initial_probs.copy()
                )

            # Anti-repetition: reduce probability of recently used words/states
            if anti_repetition and len(recent_words) >= context_window:
                recent_set = set(recent_words[-context_window:])
                for state in list(transition_probs.keys()):
                    # Penalize states that would generate recently used words
                    if state[1] in recent_set:
                        transition_probs[state] *= 0.1  # Strong penalty
                    # Extra penalty for exact state repetition
                    if state in [
                        (recent_words[i], recent_words[i + 1])
                        for i in range(
                            max(0, len(recent_words) - context_window),
                            len(recent_words) - 1,
                        )
                    ]:
                        transition_probs[state] *= 0.01

            # Sample next state
            if transition_probs:
                next_state = self._sample_from_dict(transition_probs, deterministic)

                # Verify valid 2nd-order transition
                if next_state[0] == current_state[1]:
                    new_word = next_state[1]
                    current_state = next_state
                else:
                    # Fallback for invalid transition
                    valid_states = self._find_valid_next_states(current_state[1])
                    if valid_states:
                        next_state = self._sample_from_dict(
                            valid_states, False
                        )  # Force random for variety
                        new_word = next_state[1]
                        current_state = next_state
                    else:
                        # Last resort
                        next_state = self._sample_from_dict(self._initial_probs, False)
                        new_word = next_state[1]
                        current_state = (current_state[1], new_word)
            else:
                # No valid transitions, reset
                next_state = self._sample_from_dict(self._initial_probs, False)
                new_word = next_state[1]
                current_state = (current_state[1], new_word)

            words_generated.append(new_word)
            recent_words.append(new_word)

            # Keep recent_words list manageable
            if len(recent_words) > context_window * 2:
                recent_words = recent_words[-context_window:]

        result = " ".join(words_generated[:num_words])
        log(f"Generated {num_words}", "INFO")
        return result

    def generate_words_simple(self, num_words: int, deterministic: bool = False) -> str:
        """Simplified generation method - just calls generate_words."""
        return self.generate_words(num_words, deterministic)

    def get_vocabulary_stats(self) -> Dict[str, any]:
        """Get model statistics."""
        return {
            "num_states": len(self._all_states),
            "vocabulary_size": len(self._vocab),
            "initial_states": len(self._initial_probs),
            "transition_states": len(self._transitions),
            "memory_efficient": True,
            "max_states_limit": self.max_states,
            "total_transitions": sum(len(t) for t in self._transitions.values()),
        }
