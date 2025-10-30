import argparse
import sys
from pathlib import Path

from src.ai_console_app.config import load_environment
from src.ai_console_app.keywords_openai import OpenAIKeywords
from src.ai_console_app.quiz_openai import OpenAIQuiz
from src.ai_console_app.summarizer_hf import HuggingFaceSummarizer


def read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        raise ValueError("Input file is empty")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a text file using Hugging Face, extract keywords and generate a quiz using OpenAI."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to input .txt file")
    parser.add_argument("--keywords", type=int, default=5, help="Number of keywords to generate")
    parser.add_argument("--questions", type=int, default=3, help="Number of quiz questions to generate")
    parser.add_argument("--hf-model", default="facebook/bart-large-cnn", help="Hugging Face model for summarization")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for generation")
    parser.add_argument("--dotenv", default=None, help="Optional path to a .env file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        load_environment(args.dotenv)

        text = read_text_file(Path(args.input_file))

        summarizer = HuggingFaceSummarizer(model=args.hf_model)
        summary = summarizer.summarize(text, max_words=150)

        kw = OpenAIKeywords(model=args.openai_model)
        keywords = kw.generate(text, num_keywords=args.keywords)

        quiz = OpenAIQuiz(model=args.openai_model)
        questions = quiz.generate(text, num_questions=args.questions)

        print("\n=== SUMMARY ===\n")
        print(summary)

        print("\n=== KEYWORDS ===\n")
        for i, k in enumerate(keywords, 1):
            print(f"{i}. {k}")

        print("\n=== QUIZ ===\n")
        for i, q in enumerate(questions, 1):
            print(f"Q{i}. {q['question']}")
            for j, opt in enumerate(q["options"], 1):
                print(f"   {j}) {opt}")
            print(f"   [Correct: option {q['answer_index'] + 1}]\n")

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


