import random
from collections import defaultdict


# ============================================================
# DATASET
# ============================================================

raw_sentences = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_first_order_data(sentences):
    """
    Prepare data for the first-order model.

    The first-order model uses:

        P(X_t | X_{t-1})

    Therefore, only one <START> token is required.
    """

    data = []

    for sentence in sentences:
        tokens = sentence.lower().split()

        tokens = ["<START>"] + tokens + ["<END>"]

        data.append(tokens)

    return data


def prepare_second_order_data(sentences):
    """
    Prepare data for the second-order model.

    The second-order model uses:

        P(X_t | X_{t-2}, X_{t-1})

    Therefore, two <START> tokens are required so that
    the first real word has two previous tokens as context.
    """

    data = []

    for sentence in sentences:
        tokens = sentence.lower().split()

        tokens = ["<START>", "<START>"] + tokens + ["<END>"]

        data.append(tokens)

    return data


# ============================================================
# FIRST-ORDER MODEL
# ============================================================

class FirstOrderModel:

    def __init__(self):

        # transition_counts[current_word][next_word]
        #
        # Example:
        # transition_counts["the"]["cat"] = 3

        self.transition_counts = defaultdict(lambda: defaultdict(int))

        # probabilities[current_word][next_word]
        #
        # Example:
        # probabilities["the"]["cat"] = 0.25

        self.probabilities = defaultdict(dict)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(self, sentences):

        # Count every pair of consecutive tokens

        for sentence in sentences:

            for i in range(len(sentence) - 1):

                current_word = sentence[i]
                next_word = sentence[i + 1]

                self.transition_counts[current_word][next_word] += 1

        # Convert counts to conditional probabilities

        for current_word in self.transition_counts:

            total = sum(
                self.transition_counts[current_word].values()
            )

            for next_word in self.transition_counts[current_word]:

                count = self.transition_counts[current_word][next_word]

                self.probabilities[current_word][next_word] = (
                    count / total
                )

    # --------------------------------------------------------
    # DISPLAY COUNTS
    # --------------------------------------------------------

    def show_counts(self):

        print("\nFIRST-ORDER TRANSITION COUNTS")
        print("=" * 50)

        for current_word in self.transition_counts:

            print(f"\n{current_word}:")

            for next_word, count in self.transition_counts[
                current_word
            ].items():

                print(f"    {next_word}: {count}")

    # --------------------------------------------------------
    # DISPLAY PROBABILITIES
    # --------------------------------------------------------

    def show_probabilities(self, word):

        print(f"\nP(next word | {word})")
        print("-" * 40)

        if word not in self.probabilities:

            print("No observed transitions.")

            return

        for next_word, probability in self.probabilities[word].items():

            print(
                f"P({next_word} | {word}) = "
                f"{probability:.4f}"
            )

    # --------------------------------------------------------
    # MOST PROBABLE NEXT WORD
    # --------------------------------------------------------

    def most_probable_next(self, word):

        if word not in self.probabilities:
            return None

        return max(
            self.probabilities[word],
            key=self.probabilities[word].get
        )

    # --------------------------------------------------------
    # SAMPLE NEXT WORD
    # --------------------------------------------------------

    def sample_next(self, word):

        if word not in self.probabilities:
            return None

        words = list(self.probabilities[word].keys())

        probabilities = list(
            self.probabilities[word].values()
        )

        return random.choices(
            words,
            weights=probabilities,
            k=1
        )[0]

    # --------------------------------------------------------
    # GREEDY GENERATION
    # --------------------------------------------------------

    def generate_greedy(self, max_length=20):

        current_word = "<START>"

        generated = []

        for _ in range(max_length):

            next_word = self.most_probable_next(current_word)

            if next_word is None:
                break

            # Stop when END is reached

            if next_word == "<END>":
                break

            # START should never appear after generation begins

            if next_word == "<START>":
                break

            generated.append(next_word)

            current_word = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # SAMPLING GENERATION
    # --------------------------------------------------------

    def generate_sampled(self, max_length=20):

        current_word = "<START>"

        generated = []

        for _ in range(max_length):

            next_word = self.sample_next(current_word)

            if next_word is None:
                break

            # Stop at END

            if next_word == "<END>":
                break

            # START should never be generated as a word

            if next_word == "<START>":
                continue

            generated.append(next_word)

            current_word = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # NORMALISATION TEST
    # --------------------------------------------------------

    def check_normalisation(self):

        print("\nFIRST-ORDER NORMALISATION TEST")
        print("=" * 50)

        all_valid = True

        for word in self.probabilities:

            total = sum(
                self.probabilities[word].values()
            )

            print(
                f"{word:10s} -> {total:.6f}"
            )

            if abs(total - 1.0) > 1e-9:

                all_valid = False

        if all_valid:

            print(
                "\nAll first-order probability "
                "distributions sum to 1."
            )

        else:

            print(
                "\nERROR: Some first-order "
                "distributions do not sum to 1."
            )

        return all_valid


# ============================================================
# SECOND-ORDER MODEL
# ============================================================

class SecondOrderModel:

    def __init__(self):

        # transition_counts[(word1, word2)][next_word]
        #
        # Example:
        #
        # transition_counts[("on", "the")]["mat"] = 2

        self.transition_counts = defaultdict(
            lambda: defaultdict(int)
        )

        # probabilities[(word1, word2)][next_word]

        self.probabilities = defaultdict(dict)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(self, sentences):

        # Count triples:
        #
        # word(t-2), word(t-1) -> word(t)

        for sentence in sentences:

            for i in range(len(sentence) - 2):

                word1 = sentence[i]
                word2 = sentence[i + 1]
                next_word = sentence[i + 2]

                context = (word1, word2)

                self.transition_counts[
                    context
                ][next_word] += 1

        # Convert counts to probabilities

        for context in self.transition_counts:

            total = sum(
                self.transition_counts[context].values()
            )

            for next_word in self.transition_counts[context]:

                count = self.transition_counts[
                    context
                ][next_word]

                self.probabilities[
                    context
                ][next_word] = count / total

    # --------------------------------------------------------
    # DISPLAY COUNTS
    # --------------------------------------------------------

    def show_counts(self):

        print("\nSECOND-ORDER TRANSITION COUNTS")
        print("=" * 50)

        for context in self.transition_counts:

            print(f"\n{context}:")

            for next_word, count in self.transition_counts[
                context
            ].items():

                print(f"    {next_word}: {count}")

    # --------------------------------------------------------
    # DISPLAY PROBABILITIES
    # --------------------------------------------------------

    def show_probabilities(self, word1, word2):

        context = (word1, word2)

        print(
            f"\nP(next word | {word1}, {word2})"
        )

        print("-" * 40)

        if context not in self.probabilities:

            print("No observed transitions.")

            return

        for next_word, probability in self.probabilities[
            context
        ].items():

            print(
                f"P({next_word} | {word1}, {word2}) "
                f"= {probability:.4f}"
            )

    # --------------------------------------------------------
    # MOST PROBABLE NEXT WORD
    # --------------------------------------------------------

    def most_probable_next(self, word1, word2):

        context = (word1, word2)

        if context not in self.probabilities:
            return None

        return max(
            self.probabilities[context],
            key=self.probabilities[context].get
        )

    # --------------------------------------------------------
    # SAMPLE NEXT WORD
    # --------------------------------------------------------

    def sample_next(self, word1, word2):

        context = (word1, word2)

        if context not in self.probabilities:
            return None

        words = list(
            self.probabilities[context].keys()
        )

        probabilities = list(
            self.probabilities[context].values()
        )

        return random.choices(
            words,
            weights=probabilities,
            k=1
        )[0]

    # --------------------------------------------------------
    # GREEDY GENERATION
    # --------------------------------------------------------

    def generate_greedy(self, max_length=20):

        word1 = "<START>"
        word2 = "<START>"

        generated = []

        for _ in range(max_length):

            next_word = self.most_probable_next(
                word1,
                word2
            )

            if next_word is None:
                break

            # Stop at END

            if next_word == "<END>":
                break

            # START should never be generated

            if next_word == "<START>":
                break

            generated.append(next_word)

            # Move the context window forward

            word1 = word2
            word2 = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # SAMPLING GENERATION
    # --------------------------------------------------------

    def generate_sampled(self, max_length=20):

        word1 = "<START>"
        word2 = "<START>"

        generated = []

        for _ in range(max_length):

            next_word = self.sample_next(
                word1,
                word2
            )

            if next_word is None:
                break

            # Stop at END

            if next_word == "<END>":
                break

            # START should never be generated

            if next_word == "<START>":
                continue

            generated.append(next_word)

            # Shift context window

            word1 = word2
            word2 = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # NORMALISATION TEST
    # --------------------------------------------------------

    def check_normalisation(self):

        print("\nSECOND-ORDER NORMALISATION TEST")
        print("=" * 50)

        all_valid = True

        for context in self.probabilities:

            total = sum(
                self.probabilities[context].values()
            )

            print(
                f"{context} -> {total:.6f}"
            )

            if abs(total - 1.0) > 1e-9:

                all_valid = False

        if all_valid:

            print(
                "\nAll second-order probability "
                "distributions sum to 1."
            )

        else:

            print(
                "\nERROR: Some second-order "
                "distributions do not sum to 1."
            )

        return all_valid


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # ========================================================
    # PREPARE DATA
    # ========================================================

    first_order_data = prepare_first_order_data(
        raw_sentences
    )

    second_order_data = prepare_second_order_data(
        raw_sentences
    )

    # --------------------------------------------------------
    # Display training data
    # --------------------------------------------------------

    print("\nTRAINING DATA")
    print("=" * 50)

    for sentence in first_order_data:

        print(" ".join(sentence))

    # ========================================================
    # FIRST-ORDER MODEL
    # ========================================================

    first_order = FirstOrderModel()

    first_order.train(first_order_data)

    # --------------------------------------------------------
    # Transition counts
    # --------------------------------------------------------

    first_order.show_counts()

    # --------------------------------------------------------
    # Selected probability tables
    # --------------------------------------------------------

    print("\n\nSELECTED FIRST-ORDER PROBABILITY TABLES")
    print("=" * 50)

    selected_words = [
        "the",
        "cat",
        "dog",
        "sat",
        "ran"
    ]

    for word in selected_words:

        first_order.show_probabilities(word)

    # --------------------------------------------------------
    # Most probable predictions
    # --------------------------------------------------------

    print("\n\nMOST PROBABLE NEXT WORD")
    print("=" * 50)

    for word in selected_words:

        prediction = first_order.most_probable_next(
            word
        )

        print(
            f"After '{word}' -> '{prediction}'"
        )

    # --------------------------------------------------------
    # Normalisation test
    # --------------------------------------------------------

    first_order.check_normalisation()

    # ========================================================
    # FIRST-ORDER: 20 SAMPLED SENTENCES
    # ========================================================

    print("\n\n20 SAMPLED FIRST-ORDER SENTENCES")
    print("=" * 50)

    for i in range(20):

        sentence = first_order.generate_sampled()

        print(
            f"{i + 1:2d}. {sentence}"
        )

    # ========================================================
    # FIRST-ORDER: GREEDY VS SAMPLING
    # ========================================================

    print("\n\nFIRST-ORDER GREEDY GENERATION")
    print("=" * 50)

    for i in range(5):

        sentence = first_order.generate_greedy()

        print(
            f"{i + 1}. {sentence}"
        )

    print("\n\nFIRST-ORDER SAMPLING GENERATION")
    print("=" * 50)

    for i in range(5):

        sentence = first_order.generate_sampled()

        print(
            f"{i + 1}. {sentence}"
        )

    # ========================================================
    # SECOND-ORDER MODEL
    # ========================================================

    second_order = SecondOrderModel()

    second_order.train(second_order_data)

    # --------------------------------------------------------
    # Transition counts
    # --------------------------------------------------------

    second_order.show_counts()

    # --------------------------------------------------------
    # Selected probability tables
    # --------------------------------------------------------

    print("\n\nSELECTED SECOND-ORDER PROBABILITY TABLES")
    print("=" * 50)

    selected_contexts = [

        ("<START>", "the"),

        ("the", "cat"),

        ("the", "dog"),

        ("cat", "sat"),

        ("dog", "sat"),

        ("cat", "ran"),

        ("dog", "ran"),

        ("sat", "on"),

        ("on", "the"),

        ("ran", "to")
    ]

    for word1, word2 in selected_contexts:

        second_order.show_probabilities(
            word1,
            word2
        )

    # --------------------------------------------------------
    # Normalisation test
    # --------------------------------------------------------

    second_order.check_normalisation()

    # ========================================================
    # SECOND-ORDER GREEDY GENERATION
    # ========================================================

    print("\n\nSECOND-ORDER GREEDY GENERATION")
    print("=" * 50)

    for i in range(5):

        sentence = second_order.generate_greedy()

        print(
            f"{i + 1}. {sentence}"
        )

    # ========================================================
    # SECOND-ORDER SAMPLING GENERATION
    # ========================================================

    print("\n\nSECOND-ORDER SAMPLING GENERATION")
    print("=" * 50)

    for i in range(5):

        sentence = second_order.generate_sampled()

        print(
            f"{i + 1}. {sentence}"
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()