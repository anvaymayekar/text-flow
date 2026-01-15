from __future__ import annotations
import os
import time
import math
import pickle
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from utils import log
from utils.constants import _getconst


class TrainerError(Exception):
    pass


class Trainer:
    """Word-level Markov trainer (bigram-word states)"""

    def __init__(
        self,
        chunk_bytes: int = 64 * 1024,
        use_tqdm: bool = True,
    ):
        self.corpus_path: Path = Path(_getconst("corpus"))
        self.params_path: Path = Path(_getconst("params"))
        self.chunk_bytes = chunk_bytes
        self.use_tqdm = use_tqdm
        self.initial_file = Path(_getconst("initial")) if _getconst("initial") else None
        self.transitions_file = (
            Path(_getconst("transitions")) if _getconst("transitions") else None
        )

        # runtime accumulators
        self._processed_bytes: int = 0
        self._processed_words: int = 0
        self._processed_lines: int = 0

        # word-level counts (sparse dictionaries)
        self.initial_counts: Dict[Tuple[str, str], int] = {}
        self.transition_counts: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = {}

        # loaded yaml state
        self._yaml_state: Dict = {}
        self._cursor_start: int = 0

        if not self.corpus_path.exists():
            raise TrainerError(f"corpus not found at {self.corpus_path}")

        file_size = self.corpus_path.stat().st_size

        self._file_size: Optional[int] = file_size

        # optional tqdm
        self._tqdm = None
        if self.use_tqdm:
            try:
                from tqdm import tqdm

                self._tqdm = tqdm
            except Exception:
                log(
                    "tqdm requested but not available; falling back to simple progress",
                    "WARNING",
                )
                self._tqdm = None

    # ---------------- yaml helpers ----------------
    def _load_yaml(self):
        if not self.params_path.exists():
            raise TrainerError(f"params yaml not found at {self.params_path}")
        with open(self.params_path, "r", encoding="utf8") as fh:
            data = yaml.safe_load(fh)
        if "corpus" not in data:
            raise TrainerError("params yaml missing top-level 'corpus' key")
        for k in ("cursor", "line", "count"):
            if k not in data["corpus"]:
                raise TrainerError(f"params yaml corpus missing key '{k}'")
        self._yaml_state = data
        self._cursor_start = int(data["corpus"]["cursor"])
        log(
            f"loaded yaml cursor={self._cursor_start} line={data['corpus']['line']} count={data['corpus']['count']}",
            "DEBUG",
        )

    def _write_yaml_atomic(self, new_cursor, new_line, new_count):
        corpus = self._yaml_state.setdefault("corpus", {})
        corpus["cursor"] = int(new_cursor)
        corpus["line"] = int(new_line)
        corpus["count"] = int(new_count)
        tmp_path = self.params_path.with_suffix(self.params_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf8") as fh:
            yaml.safe_dump(self._yaml_state, fh, sort_keys=False)
        os.replace(tmp_path, self.params_path)
        log("wrote updated params yaml atomically", "INFO")

    # ---------------- file helpers ----------------
    def _ensure_parent(self, path: Path):
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _load_pickle_or_empty(self, path: Optional[Path]):
        if path is None or not path.exists():
            return {}
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as e:
            log(f"failed to load pickle {path}: {e}, returning empty dict", "WARNING")
            return {}

    def _save_pickle_atomic(self, path: Path, obj):
        if path is None:
            raise TrainerError("no path configured for pickle write")
        self._ensure_parent(path)
        tmp = path.with_suffix(".tmp.pkl")
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh)
        os.replace(tmp, path)
        log(f"wrote pickle atomically to {path}", "DEBUG")

    # ---------------- core processing ----------------
    def _process_stream(self, start_byte: int):
        if not self.corpus_path.exists():
            raise TrainerError(f"corpus file not found at {self.corpus_path}")
        self._file_size = self.corpus_path.stat().st_size

        prev_prev: Optional[str] = None
        prev: Optional[str] = None
        bytes_read = 0
        start_time = time.time()

        use_tqdm = False
        pbar = None
        if self._tqdm is not None:
            try:
                remaining = max(0, self._file_size - start_byte)
                pbar = self._tqdm(
                    total=remaining, unit="B", unit_scale=True, desc="training"
                )
                use_tqdm = True
            except Exception:
                pbar = None
                use_tqdm = False

        with open(self.corpus_path, "r", encoding="utf8") as fh:
            fh.seek(start_byte)
            for line in fh:
                self._processed_lines += 1
                words = [w.lower() for w in line.strip().split() if w.isalpha()]
                for word in words:
                    if prev is None:
                        prev = word
                    elif prev_prev is None:
                        # first bigram
                        self.initial_counts[(prev, word)] = (
                            self.initial_counts.get((prev, word), 0) + 1
                        )
                        prev_prev = prev
                        prev = word
                    else:
                        from_key = (prev_prev, prev)
                        to_key = (prev, word)
                        if from_key not in self.transition_counts:
                            self.transition_counts[from_key] = {}
                        self.transition_counts[from_key][to_key] = (
                            self.transition_counts[from_key].get(to_key, 0) + 1
                        )
                        prev_prev = prev
                        prev = word
                    self._processed_words += 1

                if use_tqdm and pbar is not None:
                    pbar.update(len(line.encode("utf8")))

        if pbar is not None:
            pbar.close()

    def _validate_word_probabilities(self, initial: dict, transitions: dict) -> None:
        """Validate word-level Markov counts converted to probabilities."""

        # ---------------- initial bigram probabilities ----------------
        total_initial = sum(initial.values())
        if total_initial > 0:
            pi_sum = sum(v / total_initial for v in initial.values())
            if not abs(pi_sum - 1.0) < 1e-6:
                log(
                    f"[WORD MODEL] Initial probabilities sum to {pi_sum} (not 1)",
                    "WARNING",
                )
            else:
                log("[WORD MODEL] Initial probabilities valid", "INFO")
        else:
            log("[WORD MODEL] Initial counts empty, nothing to validate", "INFO")

        # ---------------- transition probabilities ----------------
        bad_rows = 0
        for from_key, to_dict in transitions.items():
            row_sum = sum(to_dict.values())
            if row_sum == 0:
                continue
            prob_sum = sum(cnt / row_sum for cnt in to_dict.values())
            if not abs(prob_sum - 1.0) < 1e-6:
                bad_rows += 1
                log(
                    f"[WORD MODEL] Transition row {from_key} sums to {prob_sum} (not 1)",
                    "WARNING",
                )

        if bad_rows == 0:
            log("[WORD MODEL] All transition probabilities valid", "INFO")
        else:
            log(f"[WORD MODEL] {bad_rows} transition rows invalid", "WARNING")

    # ---------------- public API ----------------
    def run_training(self):
        self._load_yaml()
        start_cursor = self._yaml_state["corpus"]["cursor"]
        existing_line = self._yaml_state["corpus"]["line"]
        existing_count = self._yaml_state["corpus"]["count"]

        if start_cursor >= self._file_size:
            log("cursor at or beyond end of file, nothing to process", "INFO")
            return

        self._process_stream(start_cursor)

        # load existing pickle counts
        existing_initial = self._load_pickle_or_empty(self.initial_file)
        existing_trans = self._load_pickle_or_empty(self.transitions_file)

        # update counts
        for k, v in self.initial_counts.items():
            existing_initial[k] = existing_initial.get(k, 0) + v
        for from_key, to_dict in self.transition_counts.items():
            if from_key not in existing_trans:
                existing_trans[from_key] = {}
            for to_key, cnt in to_dict.items():
                existing_trans[from_key][to_key] = (
                    existing_trans[from_key].get(to_key, 0) + cnt
                )

        # atomic save
        if self.initial_file:
            self._save_pickle_atomic(self.initial_file, existing_initial)
        if self.transitions_file:
            self._save_pickle_atomic(self.transitions_file, existing_trans)

        # update yaml
        new_cursor = self._file_size
        new_line = existing_line + self._processed_lines
        new_count = existing_count + self._processed_words
        self._write_yaml_atomic(new_cursor, new_line, new_count)
        self._validate_word_probabilities(existing_initial, existing_trans)
        log(
            f"training completed successfully (words {self._processed_words}, lines {self._processed_lines})",
            "INFO",
        )
