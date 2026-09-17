"""Fetch and clean public domain sentences for TTS training (zero external dependencies)."""
import argparse, re, urllib.request

SOURCES = [
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes (~3,100 sents)
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice (~5,500 sents)
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice in Wonderland (~1,500 sents)
]


def fetch_sentences(max_count=5000, out_file="sentences.txt"):
    all_sentences = []
    print(f"[*] Downloading clean sentences from open text sources...")

    for url in SOURCES:
        if len(all_sentences) >= max_count:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="ignore")

            # Remove Project Gutenberg header and footer
            start_idx = text.find("*** START OF THE PROJECT GUTENBERG")
            if start_idx != -1:
                text = text[start_idx:]
            end_idx = text.find("*** END OF THE PROJECT GUTENBERG")
            if end_idx != -1:
                text = text[:end_idx]

            # Normalize spaces and newlines
            text = re.sub(r"\r\n|\r", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            # Remove multi-newlines
            paragraphs = [p.replace("\n", " ").strip() for p in text.split("\n\n") if p.strip()]

            for p in paragraphs:
                # Split on sentence terminals: . ! ?
                raw_sents = re.split(r"(?<=[.!?])\s+", p)
                for s in raw_sents:
                    s = s.strip()
                    # Filter clean, readable sentences for TTS:
                    # - 25 to 200 characters
                    # - starts with a capital letter or quote
                    # - ends with terminal punctuation
                    # - no URLs or uppercase-only words
                    if 25 <= len(s) <= 220 and re.match(r'^[A-Z"\'“]', s) and s[-1] in ".!?\"'”":
                        # Clean quotes and underscores
                        s = s.replace("_", "").replace("--", "—").strip()
                        if s and not s.isupper():
                            all_sentences.append(s)
                            if len(all_sentences) >= max_count:
                                break
        except Exception as e:
            print(f"[!] Note: Could not fetch from {url}: {e}")

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            for s in all_sentences:
                f.write(s + "\n")
        print(f"[+] Saved {len(all_sentences):,} clean sentences to {out_file}")

    return all_sentences


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000, help="Number of sentences to fetch")
    ap.add_argument("--out", default="sentences.txt", help="Output text file")
    args = ap.parse_args()
    fetch_sentences(args.count, args.out)
