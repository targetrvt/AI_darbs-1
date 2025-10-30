from typing import Optional

from huggingface_hub import InferenceClient

from .config import try_get_env_var


class HuggingFaceSummarizer:
    def __init__(self, model: str = "facebook/bart-large-cnn", timeout: int = 60) -> None:
        token = try_get_env_var("HUGGINGFACEHUB_API_TOKEN")
        self.client = InferenceClient(token=token, timeout=timeout)
        self.model = model

    def summarize(self, text: str, max_words: int = 150) -> str:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        prompt = (
            "Summarize the following text in at most "
            f"{max_words} words. Be concise and preserve key facts.\n\n{text}\n\nSummary:"
        )

        # Prefer native summarization if available in current huggingface_hub version
        try:
            # Newer clients provide a dedicated method
            result = self.client.summarization(text, model=self.model)
            if isinstance(result, dict) and "summary_text" in result:
                return str(result["summary_text"]).strip()
            if isinstance(result, list) and result and isinstance(result[0], dict) and "summary_text" in result[0]:
                return str(result[0]["summary_text"]).strip()
        except Exception:
            # Fall back to instruction via text generation
            pass

        # Robust fallback: instruction-tuned generation
        try:
            generated = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
            )
            return str(generated).strip()
        except Exception as exc:
            raise RuntimeError(f"Hugging Face summarization failed: {exc}") from exc


