import sys
import json
import re
from io import StringIO
import pandas as pd

# we'll leverage nltk for tokenization and stopwords since that's what
# the earlier notebooks used. This keeps the style familiar and humanized.
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ensure required corpora are available; if they're already present this
# is a no-op, otherwise it'll download them the first time.
# the `punkt` tokenizer is needed for word_tokenize, and `stopwords` for
#the English stopword list.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> list[str]:
    """Convert a string into a list of word tokens.

    This wrapper simply calls NLTK's word_tokenize and lower‑cases the
    result so that the downstream logic is easier to read.
    """
    if not isinstance(text, str):
        return []
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out standard English stopwords from a list of tokens."""
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in stop_words]


def is_valid_email(email: str) -> bool:
    """Return True if the supplied string meets the simple email criteria.

    The requirements are:
      * exactly one '@'
      * at least one '.' after the '@'
      * non-empty username and domain components

    A regular expression is used to keep the logic concise.
    """
    if not isinstance(email, str):
        return False
    # this pattern checks for one @ and at least one dot after it, with
    # non-empty fragments before/after.
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None


# ---------------------------------------------------------------------------
# main processing
# ---------------------------------------------------------------------------

def main() -> None:
    # read the entire stdin as text and convert it to a DataFrame
    input_data = sys.stdin.read()
    df = pd.read_csv(StringIO(input_data))

    # make sure the columns we expect are present, otherwise the script will
    # raise a KeyError naturally which is fine for testing scenarios.

    # ------------------------------------------------------------------
    # Task 1.1 — Word Count Analysis for Harden
    # ------------------------------------------------------------------
    word_counts_before = 0
    word_counts_after = 0

    mask = df["Candidate Name"].astype(str) == "Harden"
    if mask.any():
        comment = df.loc[mask, "Extra Comment"].iloc[0]
        tokens = tokenize_text(comment)
        word_counts_before = len(tokens)
        filtered = remove_stopwords(tokens)
        word_counts_after = len(filtered)
    # if Harden isn't found the counts remain 0; tests should cover this.

    # ------------------------------------------------------------------
    # Task 1.2 — Email Validation count
    # ------------------------------------------------------------------
    valid_email_count = df["Email"].apply(is_valid_email).sum()

    # ------------------------------------------------------------------
    # Task 1.3 — Long Comment Detection
    # ------------------------------------------------------------------
    long_comment_names = []

    for _, row in df.iterrows():
        comment = row.get("Extra Comment", "")
        tokens = tokenize_text(comment)
        filtered = remove_stopwords(tokens)
        if len(filtered) > 100:
            long_comment_names.append(row["Candidate Name"])

    # sort names alphabetically as required by the spec
    long_comment_names = sorted(long_comment_names)

    # if only one name we keep it as a list as the spec states "a single name"
    # however the output format example shows always a list. we'll return the
    # list anyway; downstream tests can inspect length.

    # ------------------------------------------------------------------
    # Task 1.4 — Minimum and maximum word counts (after removing stop words)
    # ------------------------------------------------------------------
    counts_after_all = []
    for _, row in df.iterrows():
        comment = row.get("Extra Comment", "")
        tokens = tokenize_text(comment)
        filtered = remove_stopwords(tokens)
        counts_after_all.append(len(filtered))

    if counts_after_all:
        min_count = min(counts_after_all)
        max_count = max(counts_after_all)
    else:
        min_count = 0
        max_count = 0

    # build result dictionary
    result = {
        "1.1": [word_counts_before, word_counts_after],
        "1.2": int(valid_email_count),
        "1.3": long_comment_names,
        "1.4": [min_count, max_count],
    }

    # print JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
