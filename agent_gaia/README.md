---
title: GAIA Agent Final Assignment
emoji: 🕵️‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Agent

Final assignment agent for the Hugging Face Agents Course (Unit 4).

## Local run

```bash
pip install -r requirements.txt
export OPEN_ROUTER_API_KEY=your_key
export OPENROUTER_MODEL=minimax/minimax-m3
export OPENROUTER_VISION_MODEL=minimax/minimax-m3
# optional local submit without OAuth:
export HF_USERNAME=your-hf-username
export AGENT_CODE_URL=https://huggingface.co/spaces/your-username/your-space/tree/main
python app.py
```

## Hugging Face Space

1. Create a Space from this folder (Gradio SDK).
2. Add secret `OPEN_ROUTER_API_KEY`.
3. Keep the Space **public**.
4. Open the Space → **Log in with Hugging Face** → **Run Evaluation & Submit All Answers**.

Leaderboard: https://huggingface.co/spaces/agents-course/Students_leaderboard
