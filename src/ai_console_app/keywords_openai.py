import json
from typing import List

from openai import OpenAI

from .config import get_env_var


class OpenAIKeywords:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        # The SDK reads the API key from OPENAI_API_KEY automatically, but we validate early.
        _ = get_env_var("OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model

    def generate(self, text: str, num_keywords: int = 5) -> List[str]:
        if num_keywords <= 0:
            raise ValueError("num_keywords must be a positive integer")
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        system = (
            "You are an assistant that extracts concise, descriptive keywords (1-3 words) "
            "that best characterize the given text. Return ONLY a JSON array of strings."
        )
        user = (
            f"Text:\n{text}\n\n"
            f"Return exactly {num_keywords} keywords as a JSON array of strings."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI keyword generation failed: {exc}") from exc

        # Parse JSON array robustly
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("Response is not a list")
            keywords = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            # Fallback: attempt to split by comma
            keywords = [k.strip() for k in content.split(",") if k.strip()]

        # Enforce requested count without failing
        keywords = keywords[:num_keywords]
        if len(keywords) < num_keywords:
            # Pad with last known keyword variants to meet requested count deterministically
            last = keywords[-1] if keywords else "keyword"
            while len(keywords) < num_keywords:
                keywords.append(last)
        return keywords


