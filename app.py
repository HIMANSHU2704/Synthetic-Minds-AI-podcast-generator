#!/usr/bin/env python3
"""
Synthetic Minds - AI Podcast Dashboard
======================================
A beautiful Streamlit interface for generating AI-powered podcasts
with topic-driven research, Indian personas, and text-to-speech.

Run:    streamlit run app.py
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st

# ── Import our podcast engine modules ──
sys.path.insert(0, str(Path(__file__).parent))
from main_v3 import (
    init_llm,
    init_gemini,          # kept for CLI compat; app uses init_llm
    generate_research_and_personas,
    generate_intro,
    build_prompt,
    call_llm,
    call_gemini,          # alias — same as call_llm
    clean_llm_output,
    get_emotional_tone,
    get_turn_direction,
    generate_outro,
    save_transcript,
    Turn,
    ConversationState,
    Persona,
    LLM_DISPLAY_NAMES,
    MODEL,
    DEFAULT_TURNS,
    WORDS_PER_TURN,
)
from podcast import (
    parse_transcript,
    build_voice_map,
    generate_edge,
    generate_openai,
    stitch_and_master,
    _find_ffmpeg,
    _clean_text,
)


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Synthetic Minds - AI Podcast Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ── */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-header p {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0;
    }
    
    /* ── Cards ── */
    .podcast-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #2a2a4a;
    }
    .persona-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #4ECDC4;
        margin-bottom: 0.8rem;
    }
    .persona-card.host {
        border-left-color: #4ECDC4;
    }
    .persona-card.guest {
        border-left-color: #FF6B6B;
    }
    
    /* ── Transcript ── */
    .transcript-container {
        background: #16162a;
        border-radius: 12px;
        padding: 1.2rem;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #2a2a4a;
    }
    .turn-host {
        background: rgba(78, 205, 196, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #4ECDC4;
    }
    .turn-guest {
        background: rgba(255, 107, 107, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #FF6B6B;
    }
    .turn-intro {
        background: rgba(69, 183, 209, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #45B7D1;
    }
    .speaker-label {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .turn-text {
        color: #e0e0e0;
        line-height: 1.5;
        font-size: 0.92rem;
    }
    .emotion-tag {
        font-size: 0.75rem;
        color: #888;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* ── Progress ── */
    .progress-text {
        text-align: center;
        color: #4ECDC4;
        font-weight: 600;
        padding: 0.5rem 0;
    }
    
    /* ── Audio player ── */
    audio {
        width: 100%;
        border-radius: 10px;
    }
    
    /* ── Stats ── */
    .stat-box {
        text-align: center;
        padding: 0.8rem;
        background: #1e1e2f;
        border-radius: 10px;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* ── Research brief ── */
    .research-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #2a2a4a;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.6;
        font-size: 0.92rem;
    }
    
    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Keep the sidebar expand/collapse toggle always visible ── */
    /* (the ">" button lives inside <header> in newer Streamlit)  */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block   !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # ── transcript state ──
        "research": "",
        "host": None,
        "guest": None,
        "transcript_turns": [],
        "transcript_path": None,
        "audio_path": None,
        "generating": False,
        "generation_stage": "",
        "total_turns": DEFAULT_TURNS,
        "model": None,
        # ── LLM selection ──
        "llm_type": "gemini",
        # Gemini / Vertex AI Gemini
        "gemini_api_key":    os.environ.get("GEMINI_API_KEY", ""),
        "vertexai_api_key":  os.environ.get("VERTEXAI_API_KEY", ""),
        "vertexai_model":    "gemini-2.5-pro",
        # OpenAI (transcript)
        "openai_transcript_key":   os.environ.get("OPENAI_API_KEY", ""),
        "openai_transcript_model": "gpt-4o-mini",
        # Azure OpenAI
        "azure_api_key":     os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "azure_endpoint":    os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "azure_deployment":  os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        "azure_api_version": "2024-02-01",
        # Azure DeepSeek
        "azure_deepseek_endpoint": os.environ.get("AZURE_DEEPSEEK_ENDPOINT", ""),
        "azure_deepseek_token":    os.environ.get("AZURE_DEEPSEEK_TOKEN", ""),
        # TTS (OpenAI HD voices)
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def _has_transcript_credentials() -> bool:
    """Return True if the active LLM provider has the minimum required credentials."""
    lt = st.session_state.get("llm_type", "gemini")
    if lt == "gemini":
        return bool(st.session_state.get("gemini_api_key"))
    elif lt == "openai":
        return bool(st.session_state.get("openai_transcript_key"))
    elif lt == "azure-openai":
        return all([
            st.session_state.get("azure_api_key"),
            st.session_state.get("azure_endpoint"),
            st.session_state.get("azure_deployment"),
        ])
    elif lt == "azure-deepseek":
        return all([
            st.session_state.get("azure_deepseek_endpoint"),
            st.session_state.get("azure_deepseek_token"),
        ])
    elif lt == "vertexai-gemini":
        return bool(st.session_state.get("vertexai_api_key"))
    return False


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # ── Transcript LLM ──
    st.markdown("**Transcript LLM**")

    llm_options      = list(LLM_DISPLAY_NAMES.keys())
    llm_labels       = list(LLM_DISPLAY_NAMES.values())
    current_llm_idx  = llm_options.index(st.session_state.llm_type) \
                       if st.session_state.llm_type in llm_options else 0

    selected_label = st.selectbox(
        "LLM Provider",
        options=llm_labels,
        index=current_llm_idx,
        help="Which LLM to use for transcript generation (research, personas, conversation).",
        label_visibility="collapsed",
    )
    st.session_state.llm_type = llm_options[llm_labels.index(selected_label)]

    # ── Credential inputs — shown only for the active provider ──
    lt = st.session_state.llm_type

    if lt == "gemini":
        gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="Enter your Gemini API key...",
            help="Get one at makersuite.google.com",
        )
        if gemini_key:
            st.session_state.gemini_api_key = gemini_key
            os.environ["GEMINI_API_KEY"] = gemini_key

    elif lt == "openai":
        oai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_transcript_key,
            type="password",
            placeholder="sk-...",
        )
        if oai_key:
            st.session_state.openai_transcript_key = oai_key
        st.session_state.openai_transcript_model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"].index(
                st.session_state.openai_transcript_model
            ),
        )

    elif lt == "azure-openai":
        az_key = st.text_input(
            "Azure API Key",
            value=st.session_state.azure_api_key,
            type="password",
        )
        az_endpoint = st.text_input(
            "Azure Endpoint",
            value=st.session_state.azure_endpoint,
            placeholder="https://<resource>.openai.azure.com/",
        )
        az_deployment = st.text_input(
            "Deployment Name",
            value=st.session_state.azure_deployment,
            placeholder="my-gpt4o-deployment",
        )
        az_api_version = st.text_input(
            "API Version",
            value=st.session_state.azure_api_version,
            placeholder="2024-02-01",
        )
        if az_key:        st.session_state.azure_api_key     = az_key
        if az_endpoint:   st.session_state.azure_endpoint    = az_endpoint
        if az_deployment: st.session_state.azure_deployment  = az_deployment
        if az_api_version: st.session_state.azure_api_version = az_api_version

    elif lt == "azure-deepseek":
        ds_endpoint = st.text_input(
            "Azure Endpoint",
            value=st.session_state.azure_deepseek_endpoint,
            placeholder="https://<resource>.services.ai.azure.com/...",
        )
        ds_token = st.text_input(
            "Azure Token",
            value=st.session_state.azure_deepseek_token,
            type="password",
        )
        if ds_endpoint: st.session_state.azure_deepseek_endpoint = ds_endpoint
        if ds_token:    st.session_state.azure_deepseek_token    = ds_token

    elif lt == "vertexai-gemini":
        vx_key = st.text_input(
            "API Key",
            value=st.session_state.vertexai_api_key,
            type="password",
            help="Vertex AI API key from Google Cloud Console.",
        )
        vx_model = st.text_input(
            "Model",
            value=st.session_state.vertexai_model,
            placeholder="gemini-2.5-pro",
        )
        if vx_key:   st.session_state.vertexai_api_key = vx_key
        if vx_model: st.session_state.vertexai_model   = vx_model

    # ── Credential status indicator ──
    if _has_transcript_credentials():
        st.success(f"{LLM_DISPLAY_NAMES[lt]} credentials set ✓")
    else:
        st.warning(f"{LLM_DISPLAY_NAMES[lt]} credentials missing")

    st.markdown("---")

    # ── Generation Settings ──
    st.markdown("**Generation Settings**")
    total_turns = st.slider(
        "Conversation Turns",
        min_value=10,
        max_value=40,
        value=DEFAULT_TURNS,
        step=2,
        help="Number of back-and-forth exchanges. 26 turns ~ 10 minutes.",
    )
    st.session_state.total_turns = total_turns

    tts_engine = st.selectbox(
        "TTS Engine",
        options=["edge (free)", "openai (paid, HD)"],
        index=0,
        help="Edge TTS is free and requires no API key. OpenAI TTS is higher quality but requires an API key.",
    )
    use_indian_voices = st.toggle(
        "Use Indian-Accented Voices",
        value=True,
        help="Use en-IN voices for authentic Indian podcast feel (Edge TTS only).",
    )

    st.markdown("---")

    # ── OpenAI TTS key (separate from transcript key) ──
    if "openai" in tts_engine:
        openai_key = st.text_input(
            "OpenAI API Key (for TTS)",
            value=st.session_state.openai_api_key,
            type="password",
            placeholder="sk-... required for OpenAI HD TTS",
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
        st.markdown("---")

    # ── System Info ──
    st.markdown("**System Status**")
    ffmpeg_ok, _ = _find_ffmpeg()
    if ffmpeg_ok:
        st.success("FFmpeg detected")
    else:
        st.error("FFmpeg not found")
        st.info("Install: `brew install ffmpeg` or `sudo apt install ffmpeg`")

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#666; font-size:0.8rem;'>Synthetic Minds v3.0</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <h1>🎙️ Synthetic Minds</h1>
    <p>AI-Powered Podcast Generator — Research, Personas & Voice</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════
#  TOPIC INPUT SECTION
# ══════════════════════════════════════════════════════════════

st.markdown("### 📝 Enter Your Topic")

col1, col2 = st.columns([4, 1])

with col1:
    topic = st.text_input(
        "Podcast Topic",
        placeholder="e.g., The future of AI in Indian healthcare, Startup culture in Bangalore, Mental health among Indian youth...",
        label_visibility="collapsed",
    )

with col2:
    generate_btn = st.button(
        "🎙️ Generate Podcast",
        type="primary",
        use_container_width=True,
        disabled=not topic or not _has_transcript_credentials(),
    )


# ══════════════════════════════════════════════════════════════
#  GENERATION LOGIC
# ══════════════════════════════════════════════════════════════

def _is_intro_turn(t):
    """Check if a turn is part of the introduction (has welcome/intro emotion)."""
    emotion = getattr(t, 'emotion', '').lower()
    return 'welcome' in emotion or 'setting the stage' in emotion or 'introduc' in emotion


def _render_live_transcript(turns, host_name, guest_name):
    """Render transcript turns for the live display."""
    html = "<div class='transcript-container'>"
    for t in turns:
        if _is_intro_turn(t):
            css_class = "turn-intro"
            speaker_color = "#45B7D1"
        elif t.speaker == host_name:
            css_class = "turn-host"
            speaker_color = "#4ECDC4"
        else:
            css_class = "turn-guest"
            speaker_color = "#FF6B6B"
        html += f"""
        <div class='{css_class}'>
            <div class="speaker-label" style="color:{speaker_color}">{t.speaker}</div>
            <div class="turn-text">{t.text}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

if generate_btn and topic and _has_transcript_credentials():
    st.session_state.generating = True
    st.session_state.generation_stage = "Initializing..."
    st.session_state.transcript_turns = []
    st.session_state.transcript_path = None
    st.session_state.audio_path = None
    st.session_state.research = ""
    st.session_state.host = None
    st.session_state.guest = None
    
    # ── Progress container ──
    progress_container = st.container()
    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        def update_status(stage: str, pct: float):
            status_text.markdown(f"<div class='progress-text'>{stage}</div>", unsafe_allow_html=True)
            progress_bar.progress(min(int(pct), 100))
    
    # ── STEP 1: Initialize Model ──
    _lt = st.session_state.llm_type
    update_status(f"Step 1/4: Initializing {LLM_DISPLAY_NAMES[_lt]}...", 5)
    try:
        if _lt == "gemini":
            model = init_llm("gemini", api_key=st.session_state.gemini_api_key)
        elif _lt == "openai":
            model = init_llm("openai",
                             api_key=st.session_state.openai_transcript_key,
                             model_name=st.session_state.openai_transcript_model)
        elif _lt == "azure-openai":
            model = init_llm("azure-openai",
                             api_key=st.session_state.azure_api_key,
                             endpoint=st.session_state.azure_endpoint,
                             deployment=st.session_state.azure_deployment,
                             api_version=st.session_state.azure_api_version)
        elif _lt == "azure-deepseek":
            model = init_llm("azure-deepseek",
                             endpoint=st.session_state.azure_deepseek_endpoint,
                             token=st.session_state.azure_deepseek_token)
        elif _lt == "vertexai-gemini":
            model = init_llm("vertexai-gemini",
                             api_key=st.session_state.vertexai_api_key,
                             model_name=st.session_state.vertexai_model)
        st.session_state.model = model
    except Exception as e:
        st.error(f"Failed to initialize {LLM_DISPLAY_NAMES[_lt]}: {e}")
        st.session_state.generating = False
        st.stop()
    
    # ── STEP 2: Research & Personas ──
    update_status("Step 2/4: Deep research & creating Indian personas...", 15)
    time.sleep(0.5)
    
    try:
        research, host, guest = generate_research_and_personas(model, topic)
        st.session_state.research = research
        st.session_state.host = host
        st.session_state.guest = guest
    except Exception as e:
        st.error(f"Failed to generate research & personas: {e}")
        st.session_state.generating = False
        st.stop()
    
    update_status("Step 2/4: Research & personas ready!", 25)
    time.sleep(0.3)
    
    # ── STEP 3: Introduction ──
    update_status("Step 3/4: Generating podcast introduction...", 30)
    
    try:
        intro_turns = generate_intro(model, topic, research, host, guest)
        st.session_state.transcript_turns.extend(intro_turns)
    except Exception as e:
        st.error(f"Failed to generate intro: {e}")
        st.session_state.generating = False
        st.stop()
    
    update_status("Step 3/4: Introduction complete! Starting conversation...", 35)
    time.sleep(0.3)
    
    # ── STEP 4: Main Conversation ──
    state = ConversationState(
        topic=topic,
        research=research,
        host=host,
        guest=guest,
        personal_story_count={host.name: 0, guest.name: 0},
    )
    # Add intro turns to state history
    state.history = list(intro_turns)
    
    agents = [host, guest]
    total = st.session_state.total_turns
    
    # Create a live transcript display area
    transcript_live = st.empty()
    
    for turn_index in range(total):
        speaker = agents[turn_index % 2]
        other = agents[(turn_index + 1) % 2]
        emotion = get_emotional_tone(turn_index)
        
        pct = 35 + int((turn_index / total) * 55)
        update_status(
            f"Step 4/4: Generating conversation... Turn {turn_index + 1}/{total} ({speaker.name} speaking)",
            pct,
        )
        
        prompt = build_prompt(speaker, other, topic, research, state, turn_index, total)
        utterance = call_llm(model, prompt)
        
        turn = Turn(speaker=speaker.name, text=utterance, emotion=emotion, turn_index=turn_index)
        state.history.append(turn)
        st.session_state.transcript_turns.append(turn)
        
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
        
        # Update live transcript display
        with transcript_live.container():
            _render_live_transcript(st.session_state.transcript_turns, host.name, guest.name)
        
        time.sleep(0.2)
    
    # ═════════════════════════════════════════════════════════
    # ── STEP 5: Generate Outro ──
    # ═════════════════════════════════════════════════════════
    update_status("Step 5/5: Host is wrapping up the show...", 90)
    
    try:
        outro_turn = generate_outro(model, state)
        
        # Append to backend state (for saving) and UI state (for display)
        state.history.append(outro_turn)
        st.session_state.transcript_turns.append(outro_turn)
        
        # Render the final text in the UI
        with transcript_live.container():
            _render_live_transcript(st.session_state.transcript_turns, host.name, guest.name)
            
    except Exception as e:
        st.error(f"Failed to generate outro: {e}")
        st.session_state.generating = False
        st.stop()

    # ── Save Transcript ──
    update_status("Saving transcript...", 92)
    transcript_path, total_words, host_words, guest_words = save_transcript(topic, state, total)
    st.session_state.transcript_path = transcript_path
    
    # Store stats
    st.session_state.total_words = total_words
    st.session_state.host_words = host_words
    st.session_state.guest_words = guest_words
    
    update_status("Podcast generation complete!", 100)
    time.sleep(0.5)
    
    st.session_state.generating = False
    status_text.empty()
    progress_bar.empty()
    transcript_live.empty()
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  DISPLAY: RESEARCH & PERSONAS
# ══════════════════════════════════════════════════════════════

if st.session_state.research and st.session_state.host and st.session_state.guest:
    st.markdown("---")
    st.markdown("### 🔬 Research & Personas")
    
    # -- Research Brief --
    with st.expander("📋 View Research Brief", expanded=False):
        st.markdown(f"<div class='research-box'>{st.session_state.research}</div>", unsafe_allow_html=True)
    
    # -- Personas Side by Side --
    pcol1, pcol2 = st.columns(2)
    
    host = st.session_state.host
    guest = st.session_state.guest
    
    with pcol1:
        st.markdown(f"""
        <div class="persona-card host">
            <h4>🎤 {host.name} — {host.role}</h4>
            <p><strong>{host.credentials}</strong> from {host.city}</p>
            <p><em>{host.archetype}</em></p>
            <p>{host.backstory}</p>
            <p><strong>Believes:</strong> {host.core_belief}</p>
            <p><strong>Humor:</strong> {host.humor_style}</p>
            <p><strong>Hot buttons:</strong> {', '.join(host.hot_buttons)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pcol2:
        st.markdown(f"""
        <div class="persona-card guest">
            <h4>🎧 {guest.name} — {guest.role}</h4>
            <p><strong>{guest.credentials}</strong> from {guest.city}</p>
            <p><em>{guest.archetype}</em></p>
            <p>{guest.backstory}</p>
            <p><strong>Believes:</strong> {guest.core_belief}</p>
            <p><strong>Humor:</strong> {guest.humor_style}</p>
            <p><strong>Hot buttons:</strong> {', '.join(guest.hot_buttons)}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DISPLAY: STATS
# ══════════════════════════════════════════════════════════════

if st.session_state.transcript_turns:
    st.markdown("---")
    st.markdown("### 📊 Episode Stats")
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    
    total_turns_display = len(st.session_state.transcript_turns)
    total_words = getattr(st.session_state, 'total_words', sum(len(t.text.split()) for t in st.session_state.transcript_turns))
    est_duration = f"{total_words / 150:.1f}"
    
    with scol1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_turns_display}</div>
            <div class="stat-label">Total Turns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with scol2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_words:,}</div>
            <div class="stat-label">Total Words</div>
        </div>
        """, unsafe_allow_html=True)
    
    with scol3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{est_duration}m</div>
            <div class="stat-label">Est. Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with scol4:
        host = st.session_state.host
        host_pct = getattr(st.session_state, 'host_words', 0) * 100 // max(total_words, 1)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{host_pct}% / {100-host_pct}%</div>
            <div class="stat-label">{host.name if host else 'Host'} / {st.session_state.guest.name if st.session_state.guest else 'Guest'}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DISPLAY: TRANSCRIPT
# ══════════════════════════════════════════════════════════════

if st.session_state.transcript_turns:
    st.markdown("---")
    st.markdown("### 📜 Transcript")
    
    # Toggle for expand/collapse
    show_full = st.toggle("Show full transcript", value=True)
    
    if show_full:
        host = st.session_state.host
        guest = st.session_state.guest
        
        st.markdown("<div class='transcript-container'>", unsafe_allow_html=True)
        for t in st.session_state.transcript_turns:
            if _is_intro_turn(t):
                css_class = "turn-intro"
                speaker_color = "#45B7D1"
            elif t.speaker == host.name:
                css_class = "turn-host"
                speaker_color = "#4ECDC4"
            else:
                css_class = "turn-guest"
                speaker_color = "#FF6B6B"
            
            emotion_text = getattr(t, 'emotion', '')
            emotion_html = f'<div class="emotion-tag">{emotion_text}</div>' if emotion_text else ''
            
            st.markdown(f"""
            <div class="{css_class}">
                <div class="speaker-label" style="color:{speaker_color}">{t.speaker}</div>
                <div class="turn-text">{t.text}</div>
                {emotion_html}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download transcript button
    if st.session_state.transcript_path and os.path.exists(st.session_state.transcript_path):
        with open(st.session_state.transcript_path, "r", encoding="utf-8") as f:
            transcript_content = f.read()
        st.download_button(
            label="📥 Download Transcript",
            data=transcript_content,
            file_name="podcast_transcript.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════════
#  AUDIO GENERATION SECTION
# ══════════════════════════════════════════════════════════════

if st.session_state.transcript_path and os.path.exists(st.session_state.transcript_path):
    st.markdown("---")
    st.markdown("### 🔊 Audio Generation")
    
    # Check if audio already exists
    default_output = "podcast_output.mp3"
    
    col_a1, col_a2 = st.columns([3, 1])
    
    with col_a1:
        output_filename = st.text_input(
            "Output filename",
            value=default_output,
            placeholder="podcast_output.mp3",
        )
    
    with col_a2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_audio_btn = st.button(
            "🔊 Generate Audio",
            type="primary",
            use_container_width=True,
        )
    
    if generate_audio_btn:
        # Check ffmpeg
        ffmpeg_path, ffprobe_path = _find_ffmpeg()
        if not ffmpeg_path:
            st.error("FFmpeg not found! Please install FFmpeg to generate audio.")
            st.info("**macOS:** `brew install ffmpeg`  |  **Ubuntu:** `sudo apt install ffmpeg`  |  **Windows:** Download from ffmpeg.org")
        else:
            with st.spinner("Generating podcast audio... This may take a few minutes."):
                try:
                    # Parse transcript
                    meta, turns = parse_transcript(st.session_state.transcript_path)
                    
                    # Build voice map
                    engine = "openai" if "openai" in tts_engine else "edge"
                    voice_map = build_voice_map(
                        turns, engine,
                        use_indian_voices=use_indian_voices and engine == "edge",
                    )
                    
                    # Generate TTS
                    with tempfile.TemporaryDirectory(prefix="podcast_tts_") as tmpdir:
                        if engine == "openai":
                            if not st.session_state.openai_api_key:
                                st.error("OpenAI API key is required for OpenAI TTS. Please add it in the sidebar.")
                                st.stop()
                            seg_paths = generate_openai(turns, voice_map, tmpdir)
                        else:
                            seg_paths = generate_edge(turns, voice_map, tmpdir)
                        
                        # Stitch & master
                        output = stitch_and_master(seg_paths, turns, output_filename, meta)
                    
                    st.session_state.audio_path = output
                    st.success(f"Podcast audio generated: {output}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")
    
    # Display audio player if audio exists
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        st.markdown("### 🎧 Your Podcast")
        
        # Audio player
        with open(st.session_state.audio_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
        
        # Download button
        with open(st.session_state.audio_path, "rb") as f:
            st.download_button(
                label="📥 Download MP3",
                data=f,
                file_name=output_filename,
                mime="audio/mpeg",
            )
        
        # File info
        file_size_mb = os.path.getsize(st.session_state.audio_path) / 1e6
        st.info(f"File: `{st.session_state.audio_path}` | Size: {file_size_mb:.1f} MB | Quality: 192kbps")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#555; font-size:0.85rem;'>
    Synthetic Minds Podcast Engine v3.0 — Powered by Gemini & Edge TTS<br>
    <span style='color:#444;'>Made for the Indian audience 🇮🇳</span>
</p>
""", unsafe_allow_html=True)