import json
from typing import Dict, List

from openai import OpenAI

from .config import get_env_var


class OpenAIQuiz:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        _ = get_env_var("OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model

    def generate(self, text: str, num_questions: int = 3) -> List[Dict]:
        if num_questions <= 0:
            raise ValueError("num_questions must be a positive integer")
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        system = (
            "You generate factual multiple-choice quiz questions from the given text. "
            "Return STRICT JSON ONLY: a list of {question, options, answer_index}. "
            "Each 'options' must be an array of exactly 4 strings. 'answer_index' is 0..3."
        )
        user = (
            f"Text:\n{text}\n\n"
            f"Return exactly {num_questions} items as JSON. Do not include any explanations."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI quiz generation failed: {exc}") from exc

        try:
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("Response is not a list")
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON returned for quiz: {exc}\nRaw: {content}") from exc

        normalized: List[Dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            options = item.get("options", [])
            answer_index = item.get("answer_index", None)

            if not question or not isinstance(options, list) or len(options) != 4:
                continue
            options = [str(o).strip() for o in options]

            try:
                answer_index_int = int(answer_index)
            except Exception:
                continue
            if not 0 <= answer_index_int < 4:
                continue

            normalized.append({
                "question": question,
                "options": options,
                "answer_index": answer_index_int,
            })

        # Enforce count by truncating; if fewer than requested, return what we have
        return normalized[:num_questions]


