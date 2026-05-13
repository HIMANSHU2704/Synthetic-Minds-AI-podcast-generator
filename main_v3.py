"""
╔══════════════════════════════════════════════════════════════╗
║          SYNTHETIC MINDS — AI PODCAST ENGINE  v3             ║
║     Topic-Driven Research • Indian Personas • Host-Guest     ║
║           Intro • Flowing Conversation • ~10 min             ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    pip install google-generativeai
    export GEMINI_API_KEY="your_key_here"
    python main_v3.py
    python main_v3.py --topic "Is social media destroying democracy?"
    python main_v3.py --topic "The future of work" --turns 28
"""

import os
import sys
import time
import random
import textwrap
import argparse
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import google.generativeai as genai

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

MODEL              = "gemini-3-flash-preview"
DEFAULT_TURNS      = 26        # ~10 min at ~90 words/turn x 150 wpm speech rate
WORDS_PER_TURN     = 90        # Target spoken word count per turn (natural cadence)
HISTORY_WINDOW     = 10        # Last N turns fed as context (keeps prompts lean)
INTERRUPT_CHANCE   = 0.12      # 12% chance of a short reactive interjection
LAUGH_CHANCE       = 0.08      # 8% chance of a genuine laugh/reaction moment

# Emotional arc - how the conversation's tone evolves across turns
EMOTIONAL_ARC = [
    (0,   3,  "warm and welcoming - you're genuinely excited to dig into this topic"),
    (4,   7,  "curious and exploratory - you're thinking out loud, building ideas"),
    (8,   12, "passionate and slightly intense - this topic actually means something to you personally"),
    (13,  17, "heated but respectful - you disagree and you're not backing down, but you still like each other"),
    (18,  21, "reflective and a little vulnerable - share something personal or admit a doubt"),
    (22,  26, "warm and winding down - finding common ground, leaving listeners with something to think about"),
]


# ══════════════════════════════════════════════════════════════
#  LLM CLIENT ABSTRACTION
# ══════════════════════════════════════════════════════════════

class LLMClient(ABC):
    """Unified interface for different LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        ...


class GeminiClient(LLMClient):
    """Google Gemini via google-generativeai SDK (AI Studio API key)."""

    def __init__(self, api_key: str, model_name: str = MODEL):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text


class OpenAIClient(LLMClient):
    """OpenAI API (GPT models)."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("openai package not found — run: pip install openai")
        self._client = _OpenAI(api_key=api_key)
        self._model  = model_name

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI (GPT models hosted on Azure)."""

    def __init__(self, api_key: str, endpoint: str, deployment: str,
                 api_version: str = "2024-02-01"):
        try:
            from openai import AzureOpenAI as _AzureOpenAI
        except ImportError:
            raise ImportError("openai package not found — run: pip install openai")
        self._client     = _AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class AzureDeepSeekClient(LLMClient):
    """Azure AI DeepSeek via Azure AI Inference endpoint."""

    def __init__(self, endpoint: str, token: str):
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-ai-inference not found — run: pip install azure-ai-inference"
            )
        self._client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

    def generate(self, prompt: str) -> str:
        from azure.ai.inference.models import UserMessage
        resp = self._client.complete(messages=[UserMessage(content=prompt)])
        return resp.choices[0].message.content


class VertexAIGeminiClient(LLMClient):
    """Google Vertex AI Gemini (API-key access via google-generativeai SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text


# ── Human-readable names used in the UI ──
LLM_DISPLAY_NAMES: Dict[str, str] = {
    "gemini":          "Gemini (Google AI Studio)",
    "openai":          "OpenAI",
    "azure-openai":    "Azure OpenAI",
    "azure-deepseek":  "Azure DeepSeek",
    "vertexai-gemini": "Vertex AI Gemini",
}


def init_llm(llm_type: str, **credentials) -> LLMClient:
    """
    Factory — create an LLMClient for the given provider.

    Args:
        llm_type: One of the keys in LLM_DISPLAY_NAMES.
        **credentials: Provider-specific keyword arguments:
            gemini          → api_key, [model_name]
            openai          → api_key, [model_name]
            azure-openai    → api_key, endpoint, deployment, [api_version]
            azure-deepseek  → endpoint, token
            vertexai-gemini → api_key, [model_name]
    """
    if llm_type == "gemini":
        return GeminiClient(
            api_key=credentials["api_key"],
            model_name=credentials.get("model_name", MODEL),
        )
    elif llm_type == "openai":
        return OpenAIClient(
            api_key=credentials["api_key"],
            model_name=credentials.get("model_name", "gpt-4o-mini"),
        )
    elif llm_type == "azure-openai":
        return AzureOpenAIClient(
            api_key=credentials["api_key"],
            endpoint=credentials["endpoint"],
            deployment=credentials["deployment"],
            api_version=credentials.get("api_version", "2024-02-01"),
        )
    elif llm_type == "azure-deepseek":
        return AzureDeepSeekClient(
            endpoint=credentials["endpoint"],
            token=credentials["token"],
        )
    elif llm_type == "vertexai-gemini":
        return VertexAIGeminiClient(
            api_key=credentials["api_key"],
            model_name=credentials.get("model_name", "gemini-2.5-pro"),
        )
    else:
        raise ValueError(
            f"Unknown llm_type {llm_type!r}. Valid values: {list(LLM_DISPLAY_NAMES)}"
        )

# ══════════════════════════════════════════════════════════════
#  PERSONAS
# ══════════════════════════════════════════════════════════════

@dataclass
class Persona:
    name:            str
    role:            str
    backstory:       str          # Rich personal history that bleeds into the conversation
    archetype:       str
    core_belief:     str          # Their fundamental position on most things
    hot_buttons:     List[str]    # Topics that make them emotional
    humor_style:     str
    speech_patterns: List[str]    # Phrases, rhythms, and habits
    emotional_tells: List[str]    # How they show emotion in speech
    tendency:        str          # How they react when challenged
    credentials:     str = ""     # Professional credentials (for intro)
    city:            str = ""     # Indian city they're from


# ══════════════════════════════════════════════════════════════
#  CONVERSATION STATE
# ══════════════════════════════════════════════════════════════

@dataclass
class Turn:
    speaker:    str
    text:       str
    emotion:    str
    turn_index: int

@dataclass
class ConversationState:
    topic:                str
    research:             str = ""
    host:                 Optional[Persona] = None
    guest:                Optional[Persona] = None
    history:              List[Turn] = field(default_factory=list)
    agree_streak:         int = 0
    last_question_at:     int = -99
    personal_story_count: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  GEMINI CLIENT
# ══════════════════════════════════════════════════════════════

def init_gemini(api_key: str) -> LLMClient:
    """Convenience alias for the CLI runner — creates a GeminiClient."""
    return GeminiClient(api_key=api_key)


def clean_llm_output(text: str, strip_speaker_labels: bool = False) -> str:
    """Thoroughly clean LLM output to ensure smooth TTS and transcript parsing.
    
    Args:
        text: The raw LLM output text
        strip_speaker_labels: If True, also strip speaker name prefixes like "Arjun: "
                             Only use this when parsing intro dialogue, NOT general conversation.
    """
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text.strip())
    
    # Remove markdown formatting (bold, italic, headers, bullets, code)
    for pattern in [r'\*\*', r'__', r'##\s+', r'#\s+', r'\*\s+', r'\n-\s+', r'\n\*\s+', r'`', r'~']:
        text = re.sub(pattern, '', text)
    
    # Remove stage directions like (laughs), [pauses], *sighs* — these break TTS flow
    text = re.sub(
        r'\s*[\(\[]\s*(?:laughs?|pauses?|sighs?|chuckles?|shrugs?|pauses?\s+for\s+a\s+moment|beat|nods?|gestures?)\s*[\)\]]\s*',
        ' ', text, flags=re.IGNORECASE
    )
    
    # Only strip speaker labels for intro parsing — never for general conversation
    # as this would corrupt natural speech containing colons
    if strip_speaker_labels:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Match speaker label lines like "Arjun: some text" at the start of a line
            if re.match(r'^[A-Za-z]+:\s*\S', stripped):
                cleaned_lines.append(re.sub(r'^[A-Za-z]+:\s*', '', stripped))
            else:
                cleaned_lines.append(stripped)
        text = '\n'.join(cleaned_lines)
    
    # Normalize multiple blank lines to single
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def call_llm(model: LLMClient, prompt: str) -> str:
    try:
        text = clean_llm_output(model.generate(prompt))
        return text
    except Exception as e:
        return f"[Error generating response: {e}]"


# Backward-compat alias so existing code that imports call_gemini still works
call_gemini = call_llm


# ══════════════════════════════════════════════════════════════
#  RESEARCH & PERSONA GENERATION
# ══════════════════════════════════════════════════════════════

def generate_research_and_personas(model: LLMClient, topic: str) -> Tuple[str, Persona, Persona]:
    """Generate a research brief and two Indian personas for the given topic."""
    
    prompt = f"""You are a world-class podcast producer and researcher. Your job is to deeply research a topic and create two compelling Indian personas who would have a fascinating intellectual conversation about it.

## Topic: {topic}

## Your Task:
1. **Research Brief**: Write a comprehensive research brief on this topic (400-500 words). Include:
   - Key facts, statistics, and recent developments
   - Different perspectives and debates around the topic
   - Indian context — how this topic specifically affects or relates to India
   - Interesting angles or lesser-known facts that would spark great conversation

2. **Host Persona**: Create a compelling Indian podcast Male host who would be perfect to discuss this topic. The host should be:
   - Well-informed, curious, and a great conversationalist
   - Someone Indian audience would relate to and respect
   - Not an expert necessarily, but genuinely interested in learning
   - Warm, engaging, with a distinct personality
   
   Provide: name (single first name only, Indian), role ("Host"), city (major Indian city), backstory (3-4 sentences max), archetype, core_belief, hot_buttons (list of 3 strings), humor_style, speech_patterns (list of 6 natural phrases), emotional_tells (list of 3 strings), tendency, credentials (brief professional background, 1-2 sentences).

3. **Guest Persona**: Create an Female Indian expert or thought leader on this topic who would be a fascinating guest. The guest should be:
   - Genuinely knowledgeable with real credentials
   - Have a distinct point of view (not generic)
   - Someone with lived experience or deep expertise
   - Interesting personality with opinions and stories
   
   Provide: name (single first name only, Indian), role ("Guest"), city (Indian city), backstory (3-4 sentences max), archetype, core_belief, hot_buttons (list of 3 strings), humor_style, speech_patterns (list of 6 natural phrases), emotional_tells (list of 3 strings), tendency, credentials (brief professional background, 1-2 sentences).

## Output Format — STRICT JSON:
Return ONLY a JSON object in this exact structure. No markdown, no code fences, no explanation. Just pure JSON starting with {{ and ending with }}.

{{
    "research": "the full research brief...",
    "host": {{
        "name": "...",
        "role": "Host",
        "city": "...",
        "backstory": "...",
        "archetype": "...",
        "core_belief": "...",
        "hot_buttons": ["...", "...", "..."],
        "humor_style": "...",
        "speech_patterns": ["...", "...", "...", "...", "...", "..."],
        "emotional_tells": ["...", "...", "..."],
        "tendency": "...",
        "credentials": "..."
    }},
    "guest": {{
        "name": "...",
        "role": "Guest",
        "city": "...",
        "backstory": "...",
        "archetype": "...",
        "core_belief": "...",
        "hot_buttons": ["...", "...", "..."],
        "humor_style": "...",
        "speech_patterns": ["...", "...", "...", "...", "...", "..."],
        "emotional_tells": ["...", "...", "..."],
        "tendency": "...",
        "credentials": "..."
    }}
}}

Make both personas feel like real, complex Indian people with distinct voices. Give them authentic Indian names and Indian cities. The host should be from a major Indian city (Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune). The guest can be from anywhere in India.

IMPORTANT: Return ONLY the JSON object, nothing else. No markdown formatting, no explanation, no code blocks. Use single-word Indian names only for both host and guest."""

    text = model.generate(prompt).strip()
    
    # Clean up the response - remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    host = Persona(**data["host"])
    guest = Persona(**data["guest"])
    
    return data["research"], host, guest


# ══════════════════════════════════════════════════════════════
#  INTRO GENERATION
# ══════════════════════════════════════════════════════════════

def generate_intro(model: LLMClient, topic: str, research: str,
                   host: Persona, guest: Persona) -> List[Turn]:
    """Generate the podcast introduction: host greeting, topic intro, guest intro."""
    
    prompt = f"""You are writing the opening of an Indian podcast episode. Write ONLY the first 2-3 speaking turns.

## Topic: {topic}
## Research Brief:
{research[:800]}

## Host: {host.name}
- {host.credentials}
- From {host.city}
- Personality: {host.archetype}, {host.humor_style}
- Speech style: Uses phrases like {', '.join(host.speech_patterns[:3])}

## Guest: {guest.name}
- {guest.credentials}
- From {guest.city}
- Personality: {guest.archetype}, {guest.humor_style}
- Speech style: Uses phrases like {', '.join(guest.speech_patterns[:3])}

## Instructions:
Write exactly 2-3 turns of natural podcast opening dialogue for an INDIAN audience:

1. **Host's Opening (REQUIRED)**: {host.name} greets the listeners warmly in an Indian way (say "Namaste" or similar warm Indian greeting), introduces the podcast show "Synthetic Minds", hooks the audience with why this topic matters to Indians specifically, introduces {guest.name} with their credentials, and welcomes them warmly. Keep it around 80-100 words. Make it feel like a real Indian podcast - warm, conversational, not overly scripted.

2. **Guest's Response (REQUIRED)**: {guest.name} responds warmly, thanks {host.name} for having them, and expresses genuine excitement about discussing this topic. Add one brief personal thought about why this topic matters to them personally or professionally. Keep it around 50-70 words.

3. **Host's Transition (OPTIONAL)**: If needed, {host.name} says one brief line to transition into the main discussion - a natural "Let's get started" equivalent. Keep it under 20 words.

## CRITICAL RULES:
- Format EXACTLY like this with speaker names followed by a colon on their own line:
{host.name}:
[what they say - just the spoken words]

{guest.name}:
[what they say - just the spoken words]

- NO bullet points, NO markdown, NO stage directions like "(laughs)" or "(pauses)"
- NO headers, NO formatting like **bold** or *italic*
- Keep it conversational and warm, like real Indian podcast hosts talk
- The host CAN naturally use occasional Hindi phrases mixed with English (Hinglish) - like "Toh aaj hum baat karenge..." or "Dekho..." - but keep it mostly accessible English
- Total: 2-3 turns only
- Do NOT include any other text or explanation outside the dialogue"""

    text = clean_llm_output(model.generate(prompt), strip_speaker_labels=True)
    
    # Parse into Turn objects
    turns = []
    lines = text.split('\n')
    current_speaker = None
    current_text_lines = []
    turn_idx = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a speaker line
        speaker_match = re.match(r'^([A-Za-z]+):\s*(.*)$', line)
        if speaker_match:
            speaker_name = speaker_match.group(1)
            remainder = speaker_match.group(2)
            
            # Save previous turn if exists
            if current_speaker and current_text_lines:
                turn_text = ' '.join(current_text_lines).strip()
                if turn_text:
                    turns.append(Turn(
                        speaker=current_speaker,
                        text=turn_text,
                        emotion="warm and welcoming - setting the stage for a great conversation",
                        turn_index=turn_idx
                    ))
                    turn_idx += 1
            
            current_speaker = speaker_name
            current_text_lines = [remainder] if remainder else []
        else:
            current_text_lines.append(line)
    
    # Save last turn
    if current_speaker and current_text_lines:
        turn_text = ' '.join(current_text_lines).strip()
        if turn_text:
            turns.append(Turn(
                speaker=current_speaker,
                text=turn_text,
                emotion="warm and welcoming - setting the stage for a great conversation",
                turn_index=turn_idx
            ))
    
    return turns


# ══════════════════════════════════════════════════════════════
#  EMOTIONAL ARC LOOKUP
# ══════════════════════════════════════════════════════════════

def get_emotional_tone(turn_index: int) -> str:
    for start, end, mood in EMOTIONAL_ARC:
        if start <= turn_index <= end:
            return mood
    return EMOTIONAL_ARC[-1][2]


# ══════════════════════════════════════════════════════════════
#  DIRECTOR - decides what this turn needs
# ══════════════════════════════════════════════════════════════

def get_turn_direction(
    state: ConversationState,
    speaker: Persona,
    other: Persona,
    turn_index: int,
    total_turns: int,
) -> Tuple[str, str]:
    """
    Returns (turn_type, instruction).
    turn_type: normal | interrupt | question | personal | pushback | laugh | wrap
    """
    turns_left = total_turns - turn_index

    # -- Wrap-up zone
    if turns_left <= 2:
        return "wrap", (
            f"We are nearing the end of the podcast. Give your final, honest thought on this topic. "
            f"Acknowledge {other.name}'s perspective if it makes sense. "
            f"IMPORTANT: Do NOT say goodbye, do NOT sign off, and do NOT thank the audience. "
            f"Just state your final point naturally. The official show wrap-up will happen after this."
        )

    # -- Force a genuine question if none has been asked in a while
    if turn_index - state.last_question_at >= 5:
        return "question", (
            f"Ask {other.name} something you actually want to know. "
            f"A real question, not a debate trick. Just genuine curiosity - "
            f"like you'd ask a friend over coffee."
        )

    # -- Force pushback if they've been agreeing too long
    if state.agree_streak >= 2:
        return "pushback", (
            f"Okay you've been agreeing too much. What do you actually not buy here? "
            f"There's something bugging you - just say it. "
            f"Be honest, not mean."
        )

    # -- Spontaneous interruption
    if random.random() < INTERRUPT_CHANCE and turn_index > 2:
        return "interrupt", (
            f"You couldn't wait - jump in on what {other.name} just said. "
            f"Keep it really short, under 40 words. Like you cut them off mid-sentence. "
            f"Raw and in the moment."
        )

    # -- Personal story (humanising, earns trust)
    personal_count = state.personal_story_count.get(speaker.name, 0)
    if personal_count < 2 and random.random() < 0.25 and turn_index > 3:
        return "personal", (
            f"Tell a small thing from your own life that connects to this. "
            f"Not a big speech - just a quick moment or memory. Something real. "
            f"Then tie it back to what you were talking about."
        )

    # -- Laugh or lightness moment
    if random.random() < LAUGH_CHANCE and turn_index > 2:
        return "laugh", (
            f"Find what's a bit funny or ridiculous about what was just said. "
            f"Laugh at it, poke fun, or make a quick joke at your own expense. "
            f"Keep it light, then get back on track."
        )

    # -- Normal turn
    return "normal", (
        f"Continue the conversation naturally. React to what was just said, add your own layer, "
        f"push the thinking forward. You can agree partially, complicate the point, or introduce something new."
    )


# ══════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

def build_prompt(
    speaker:     Persona,
    other:       Persona,
    topic:       str,
    research:    str,
    state:       ConversationState,
    turn_index:  int,
    total_turns: int,
) -> str:

    turn_type, direction = get_turn_direction(state, speaker, other, turn_index, total_turns)
    emotion = get_emotional_tone(turn_index)

    recent = state.history[-HISTORY_WINDOW:]
    transcript = (
        "\n".join(f"{t.speaker}: {t.text}" for t in recent)
        if recent else "(The podcast is just starting. You haven't spoken yet.)"
    )

    patterns = ", ".join(f'"{p}"' for p in speaker.speech_patterns[:3])
    tells    = "; ".join(speaker.emotional_tells)
    
    # Add Indian audience context and research to the prompt
    indian_context = (
        "Remember, you are speaking to an INDIAN audience. Use Indian cultural references, "
        "examples from Indian cities, Indian news, Indian policies, Indian startups, or Indian "
        "historical context where natural. Keep it relatable to educated Indian urban listeners. "
        "Occasional natural Hinglish (mixing Hindi phrases with English) is welcome and adds "
        "authenticity - but keep it accessible. Avoid heavy jargon."
    )

    prompt = f"""You're {speaker.name}. You're in the middle of a casual podcast conversation with {other.name} about {topic}.

A bit about you: {speaker.backstory}

You genuinely believe: {speaker.core_belief}
What gets you fired up: {", ".join(speaker.hot_buttons)}
How you're feeling right now: {emotion}

What you should do this turn: {direction}

Here's what was just said:
{transcript}

Now say your part. Here's how to do it right:

Talk like a real person, not a presenter. Use short sentences. Say "I mean" and "like" and "you know" sometimes. 
Cut yourself off if a thought isn't finished. Use {patterns} naturally - not all at once, just when they fit.
It's okay to sound a little unsure or worked up. Real people don't always have clean sentences.

{indian_context}

Don't use any bullet points, headers or formatting. Just talk.
Don't say your own name. Don't describe what you're doing ("laughs", "pauses" etc).
Don't start with "{other.name}," - just jump into what you want to say.
Keep it around {WORDS_PER_TURN} words, give or take.

Go:"""

    return prompt


# ══════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════

COLORS = {}
RESET  = "\033[0m"
DIM    = "\033[2m"
BOLD   = "\033[1m"

def _init_colors(host_name: str, guest_name: str):
    """Initialize speaker colors dynamically."""
    global COLORS
    COLORS = {host_name: "\033[94m", guest_name: "\033[92m"}  # Blue / Green


def display_turn(turn: Turn, turn_index: int, total_turns: int):
    color   = COLORS.get(turn.speaker, "")
    progress = int((turn_index / total_turns) * 32)
    bar     = f"{DIM}[{'█' * progress}{'░' * (32 - progress)}] {turn_index}/{total_turns}{RESET}"

    print(f"\n  {bar}")
    print(f"  {color}{BOLD}{turn.speaker}{RESET}  {DIM}· {turn.emotion}{RESET}")
    print(f"  {'─' * 68}")

    # Word-wrap and print
    words    = turn.text.split()
    line     = "  "
    line_len = 0
    for word in words:
        line     += word + " "
        line_len += len(word) + 1
        if line_len > 66:
            print(line)
            line     = "  "
            line_len = 0
    if line.strip():
        print(line)
    print()

# ══════════════════════════════════════════════════════════════
#  OUTRO GENERATION
# ══════════════════════════════════════════════════════════════

def generate_outro(model: LLMClient, state: ConversationState) -> Turn:
    """Generate the final wrap-up: host summarizes and asks for engagement."""
    host = state.host
    guest = state.guest
    
    # Compile the recent conversation for context (Gemini can easily handle the full transcript length)
    transcript = "\n".join(f"{t.speaker}: {t.text}" for t in state.history)
    
    prompt = f"""You are {host.name}, the host of the Indian podcast "Synthetic Minds". 
The conversation with your guest, {guest.name}, about "{state.topic}" has just concluded.

Here is the full transcript of your conversation:
{transcript}

## Your Task:
Write the final closing monologue for the podcast episode. 
1. Briefly summarize the key takeaways or the most profound point from the conversation.
2. Thank the guest ({guest.name}) for joining and sharing their insights.
3. Explicitly ask the listeners to increase engagement (e.g., like, share, subscribe, leave a review, comment, or follow the podcast).
4. Stay in character! Use your typical speech patterns: {', '.join(host.speech_patterns[:3])}.
5. Keep it warm, engaging, and suitable for an Indian audience. Keep it around 120-150 words.

## CRITICAL RULES:
- Return ONLY the spoken words of the monologue.
- NO bullet points, NO markdown, NO stage directions like "(laughs)" or "(pauses)".
- Do not prepend your name (e.g., do not write "{host.name}:"). Just start talking as the host."""

    # Call the LLM
    utterance = clean_llm_output(model.generate(prompt))
    
    # Return as a Turn object
    return Turn(
        speaker=host.name,
        text=utterance,
        emotion="warm, reflective, and grateful - wrapping up the show",
        turn_index=len(state.history) # Add at the end
    )


# ══════════════════════════════════════════════════════════════
#  TRANSCRIPT SAVE
# ══════════════════════════════════════════════════════════════

def save_transcript(topic: str, state: ConversationState, total_turns: int):
    """Save the full transcript in the format expected by podcast.py."""
    host = state.host
    guest = state.guest
    
    total_words = sum(len(t.text.split()) for t in state.history)
    host_words  = sum(len(t.text.split()) for t in state.history if t.speaker == host.name)
    guest_words = total_words - host_words
    output_path: str = f"{topic}_transcript.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("SYNTHETIC MINDS - PODCAST TRANSCRIPT\n")
        f.write(f"Topic  : {topic}\n")
        f.write(f"Model  : {MODEL}\n")
        f.write(f"Turns  : {total_turns}\n")
        f.write(f"Words  : ~{total_words:,}\n")
        f.write(f"Host   : {host.name} ({host.archetype}) from {host.city}\n")
        f.write(f"Guest  : {guest.name} ({guest.archetype}) from {guest.city}\n")
        f.write("=" * 70 + "\n\n")
        
        # Research brief
        f.write("RESEARCH BRIEF\n")
        f.write("-" * 40 + "\n")
        f.write(textwrap.fill(state.research, width=72) + "\n\n")
        f.write("=" * 70 + "\n\n")
        
        # Transcript turns
        for t in state.history:
            f.write(f"{t.speaker}:\n")
            f.write(textwrap.fill(t.text, width=72) + "\n\n")
    
    return output_path, total_words, host_words, guest_words


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_podcast(topic: str, api_key: str, total_turns: int = DEFAULT_TURNS, 
                show_progress: bool = True) -> ConversationState:

    model = init_gemini(api_key)
    state = ConversationState(topic=topic)
    
    # ═════════════════════════════════════════════════════════
    # STEP 1: Generate Research & Personas
    # ═════════════════════════════════════════════════════════
    if show_progress:
        print("\n" + "=" * 70)
        print(f"  RESEARCHING TOPIC: {topic}")
        print(f"  {'─' * 66}")
        print("  Generating deep research and Indian personas...")
        print("=" * 70)
    
    research, host, guest = generate_research_and_personas(model, topic)
    state.research = research
    state.host = host
    state.guest = guest
    state.personal_story_count = {host.name: 0, guest.name: 0}
    
    _init_colors(host.name, guest.name)
    
    est_min = int(total_turns * WORDS_PER_TURN / 150)
    
    if show_progress:
        print(f"\n  HOST  : {host.name} - {host.credentials} ({host.archetype})")
        print(f"  GUEST : {guest.name} - {guest.credentials} ({guest.archetype})")
        print(f"  Turns : {total_turns}   (~{est_min} min at 150 wpm)")
        print(f"  Model : {MODEL}")
        print("\n" + "=" * 70)
    
    # ═════════════════════════════════════════════════════════
    # STEP 2: Generate Introduction
    # ═════════════════════════════════════════════════════════
    if show_progress:
        print(f"  GENERATING INTRODUCTION...")
        print("=" * 70)
    
    intro_turns = generate_intro(model, topic, research, host, guest)
    for i, turn in enumerate(intro_turns):
        state.history.append(turn)
        if show_progress:
            display_turn(turn, i + 1, total_turns + len(intro_turns))
            time.sleep(0.3)
    
    # ═════════════════════════════════════════════════════════
    # STEP 3: Main Conversation
    # ═════════════════════════════════════════════════════════
    agents = [host, guest]
    
    for turn_index in range(total_turns):
        speaker = agents[turn_index % 2]
        other   = agents[(turn_index + 1) % 2]
        emotion = get_emotional_tone(turn_index)
        
        if show_progress:
            print(f"\n  {DIM}Turn {turn_index + 1}: {speaker.name} is composing...{RESET}", end="\r")
        
        prompt    = build_prompt(speaker, other, topic, research, state, turn_index, total_turns)
        utterance = call_gemini(model, prompt)
        
        turn = Turn(speaker=speaker.name, text=utterance, emotion=emotion, turn_index=turn_index)
        state.history.append(turn)
        
        # Track agreement
        if any(w in utterance.lower() for w in ["exactly", "absolutely", "i agree", "you're right", "totally", "100%"]):
            state.agree_streak += 1
        else:
            state.agree_streak = 0
        
        # Track questions
        if "?" in utterance:
            state.last_question_at = turn_index
        
        # Track personal stories
        if any(t in utterance.lower() for t in ["i grew up", "my dad", "my mom", "when i was", "i remember", "i used to"]):
            state.personal_story_count[speaker.name] = state.personal_story_count.get(speaker.name, 0) + 1
        
        if show_progress:
            display_idx = len(intro_turns) + turn_index + 1
            display_turn(turn, display_idx, total_turns + len(intro_turns))
            time.sleep(0.3)
    

    # ═════════════════════════════════════════════════════════
    # STEP 4: Generate Outro (Summary & Call to Action)
    # ═════════════════════════════════════════════════════════
    if show_progress:
        print(f"\n  GENERATING OUTRO...")
        print("=" * 70)
    
    outro_turn = generate_outro(model, state)
    state.history.append(outro_turn)
    
    if show_progress:
        display_idx = len(state.history)
        display_turn(outro_turn, display_idx, display_idx)
        time.sleep(0.3)


    # -- Wrap up
    if show_progress:
        print("\n" + "=" * 70)
        print("  EPISODE COMPLETE")
        print("=" * 70)
        
        total_words = sum(len(t.text.split()) for t in state.history)
        host_words  = sum(len(t.text.split()) for t in state.history if t.speaker == host.name)
        guest_words = total_words - host_words
        
        print(f"\n  Stats")
        print(f"     Total words  : {total_words:,}  (~{total_words/150:.1f} min at 150 wpm)")
        print(f"     {host.name:12s} : {host_words:,} words  ({host_words*100//total_words}%)")
        print(f"     {guest.name:12s} : {guest_words:,} words  ({guest_words*100//total_words}%)")
        print(f"     Turns        : {len(state.history)}")
    
    # -- Save transcript
    path, total_words, host_words, guest_words = save_transcript(topic, state, total_turns)
    
    if show_progress:
        print(f"\n  Transcript saved to: {path}\n")
    
    return state


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Minds - AI Podcast Engine v3 (Gemini-powered, Indian-focused)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_v3.py
  python main_v3.py --topic "Is hustle culture slowly killing us?"
  python main_v3.py --topic "The loneliness epidemic" --turns 30
        """
    )
    parser.add_argument("--topic",   type=str, default=None, help="Podcast topic")
    parser.add_argument("--turns",   type=int, default=DEFAULT_TURNS, help=f"Number of turns (default {DEFAULT_TURNS} ~10 min)")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n  No Gemini API key found.")
        print("    Set it:  export GEMINI_API_KEY='your_key_here'")
        print("    Or:      python main_v3.py --api-key YOUR_KEY\n")
        sys.exit(1)

    topic = args.topic
    if not topic:
        print("\n" + "=" * 50)
        print("     SYNTHETIC MINDS PODCAST")
        print("=" * 50)
        topic = input("\n  What should we talk about today?\n  -> ").strip()
        if not topic:
            topic = "Whether social media has made us lonelier as a species"
            print(f"\n  Using default topic: {topic}")

    run_podcast(topic=topic, api_key=api_key, total_turns=args.turns)


if __name__ == "__main__":
    main()