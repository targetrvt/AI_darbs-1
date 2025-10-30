# AI_darbs-1

Console app for text summarization, keyword extraction, and quiz generation.

Quick start
- Create and switch to branch `console-app`:
  - `git checkout -b console-app`
- Create a virtual environment and install deps:
  - `python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt`
- Create `.env` from template `.env.example` and set your keys:
  - `HUGGINGFACEHUB_API_TOKEN=...`
  - `OPENAI_API_KEY=...`
- Run:
  - `python main.py --input-file path\\to\\text.txt --keywords 7 --questions 3`

Environment variables
- `HUGGINGFACEHUB_API_TOKEN`: Used by Hugging Face Inference API for summarization
- `OPENAI_API_KEY`: Used by OpenAI for keywords and quiz generation

CLI usage
```bash
python main.py --input-file sample.txt --keywords 5 --questions 3 \
  --hf-model facebook/bart-large-cnn --openai-model gpt-4o-mini
```

Outputs
- Summary of the provided text
- A list of N descriptive keywords
- M multiple-choice questions (4 options each) with the correct answer index

Notes
- API keys are read from `.env` and never committed.
- Handles common errors (file issues, API errors, invalid inputs) gracefully.

PR instructions
1) Commit your changes on branch `console-app`
2) Push the branch and open a Pull Request to `main`
3) In the PR description, include:
   - What was implemented (summary, keywords, quiz)
   - How to run (commands above)
   - Screenshots of successful runs and any fixed errors
4) Submit the PR URL in E-klase before the deadline
