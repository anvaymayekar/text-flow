from utils import log
from generator import Markov, GeneratorError


# deployment of the Markov generator
def main() -> None:
    try:
        # Create the generator
        markov = Markov(max_states=1000)  # Very small limit for testing

        # Get stats first to see what we have
        stats = markov.get_vocabulary_stats()

        # Try simple generation
        if stats["num_states"] > 0:
            text = markov.generate_words(100, deterministic=False)
            print(text)
        else:
            print("No states available for generation")

    except GeneratorError as e:
        log(f"Error details: {type(e).__name__}: {str(e)}", "ERROR")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
