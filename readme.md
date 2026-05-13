# 🎙️ Synthetic Minds — AI Podcast Engine

> **An end-to-end AI pipeline that turns any topic into a fully voiced Indian podcast — research, personas, conversation, and audio — in minutes.**

---

## What It Does

Synthetic Minds takes a single topic prompt and produces a complete podcast episode:

1. **Deep Research** — Generates a 400–500 word research brief with Indian context, statistics, and debate angles using an LLM of your choice.
2. **Dynamic Personas** — Auto-creates two distinct Indian personas (a host and a guest) with rich backstories, speech patterns, emotional arcs, and regional identities.
3. **Scripted Conversation** — Runs a 10–40 turn dialogue with a built-in emotional arc: warm intro → curious exploration → passionate debate → reflection → wrap-up. Includes natural interruptions, personal stories, pushback, and humor.
4. **Outro Generation** — Host wraps up with a summary and listener call-to-action.
5. **Text-to-Speech** — Converts the transcript to audio using either free Microsoft Edge TTS (Indian voices) or paid OpenAI TTS-1-HD.
6. **Audio Mastering** — Stitches segments with speaker-aware pauses, normalizes loudness, and exports a 192kbps MP3 with ID3 podcast tags.

A **Streamlit web dashboard** (`app.py`) ties the entire pipeline together with a polished UI.

---

## Project Structure

```
synthetic-minds/
├── app.py                          # Streamlit web dashboard (main UI)
├── main_v3.py                      # Core podcast engine (LLM + transcript logic)
├── podcast.py                      # TTS + audio stitching pipeline
├── samples/
│   └── Healthcare_in_India_transcript.txt   # Sample generated transcript
└── README.md
```

### File Roles

`app.py` — The Streamlit frontend. Handles sidebar config, LLM credential input, real-time transcript display, episode stats, and audio generation. Run this to use the dashboard.

`main_v3.py` — The brain. Contains the LLM client abstraction layer, persona dataclasses, conversation state management, emotional arc engine, prompt builder, intro/outro generators, and the CLI runner. Can also be run standalone without the UI.

`podcast.py` — The voice layer. Parses the saved transcript, assigns TTS voices to speakers, generates per-turn audio via Edge TTS or OpenAI, stitches everything together with `pydub`, and exports the final MP3.

---

## Supported LLM Providers

| Provider | Key Required | Notes |
|---|---|---|
| Google Gemini (AI Studio) | `GEMINI_API_KEY` | Default; uses `gemini-2.5-flash` |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini |
| Azure OpenAI | Azure credentials | Bring your own deployment |
| Azure DeepSeek | Azure endpoint + token | Via Azure AI Inference |
| Vertex AI Gemini | Vertex API key | Google Cloud hosted |

---

## Supported TTS Engines

| Engine | Cost | Quality | API Key |
|---|---|---|---|
| Microsoft Edge TTS | Free | Good — includes Indian `en-IN` voices | None |
| OpenAI TTS-1-HD | ~$0.03/1k chars | Best | `OPENAI_API_KEY` |

---

## Prerequisites

- Python 3.9+
- FFmpeg (required for audio stitching)

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/synthetic-minds.git
cd synthetic-minds

# 2. Install Python dependencies
pip install streamlit google-generativeai openai edge-tts pydub aiofiles

# Optional: for Azure DeepSeek support
pip install azure-ai-inference

# 3. Set your LLM API key (example: Gemini)
export GEMINI_API_KEY="your_key_here"
```

---

## Usage

### Option A — Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. From the sidebar:
- Select your LLM provider and paste in credentials
- Adjust conversation turns (10–40; 26 turns ≈ 10 minutes)
- Choose TTS engine and toggle Indian-accented voices
- Enter a topic and hit **🎙️ Generate Podcast**

### Option B — CLI (Headless)

```bash
# Basic run (prompts for topic interactively)
python main_v3.py

# With topic
python main_v3.py --topic "Is hustle culture slowly killing us?"

# With topic and custom turn count
python main_v3.py --topic "The future of AI in Indian healthcare" --turns 30

# Inline API key
python main_v3.py --topic "Mental health among Indian youth" --api-key YOUR_KEY
```

### Option C — Audio Only (from existing transcript)

```bash
# Free (Edge TTS, Indian voices)
python podcast.py samples/Healthcare_in_India_transcript.txt

# High quality (OpenAI TTS)
OPENAI_API_KEY=sk-... python podcast.py samples/Healthcare_in_India_transcript.txt --engine openai

# Custom output path
python podcast.py transcript.txt --engine edge --output episode_01.mp3

# Disable Indian accents
python podcast.py transcript.txt --no-indian-voices
```

---

## Sample Output

The `samples/` folder contains a real generated transcript on **Healthcare in India** featuring:

- **Host:** Riya (Mumbai) — Curious Learner archetype
- **Guest:** Aarav (Delhi) — Innovative Visionary archetype
- **Length:** 26 turns, ~2,670 words (~10 minutes audio)
- **Model:** Gemini

This is a representative example of the transcript format that `podcast.py` expects as input.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini (default LLM) |
| `OPENAI_API_KEY` | OpenAI transcript generation and/or TTS |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name |
| `AZURE_DEEPSEEK_ENDPOINT` | Azure DeepSeek endpoint |
| `AZURE_DEEPSEEK_TOKEN` | Azure DeepSeek token |
| `FFMPEG_PATH` | Override FFmpeg binary directory (optional) |

All credentials can also be entered directly in the Streamlit sidebar — no `.env` file needed for the dashboard.

---

## Transcript Format

Generated transcripts follow this structure, which `podcast.py` parses automatically:

```
SYNTHETIC MINDS - PODCAST TRANSCRIPT
Topic  : <topic>
Model  : <model>
Turns  : <n>
Words  : ~<n>
Host   : <name> (<archetype>) from <city>
Guest  : <name> (<archetype>) from <city>
======================================================================

RESEARCH BRIEF
...

======================================================================

SpeakerName:
Spoken text goes here...

SpeakerName:
Next turn goes here...
```

---

## Configuration (main_v3.py)

Key constants you can tweak at the top of `main_v3.py`:

```python
MODEL           = "gemini-2.5-flash"   # Default Gemini model
DEFAULT_TURNS   = 26                   # ~10 min at 90 words/turn
WORDS_PER_TURN  = 90                   # Target words per speaker turn
HISTORY_WINDOW  = 10                   # Turns of context fed to each prompt
INTERRUPT_CHANCE = 0.12                # Probability of a mid-turn interjection
LAUGH_CHANCE    = 0.08                 # Probability of a humor moment
```

The emotional arc is also configurable via the `EMOTIONAL_ARC` list — a sequence of `(start_turn, end_turn, mood_description)` tuples that guide the conversation's tone progression.

---

## License

MIT