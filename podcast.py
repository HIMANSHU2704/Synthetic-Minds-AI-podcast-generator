#!/usr/bin/env python3
"""
podcast_tts.py - Transcript to Voice Podcast Converter (Proof of Concept)
=========================================================================

Parses a multi-speaker podcast transcript (the SYNTHETIC MINDS format)
and converts it to a publication-ready MP3 using Text-to-Speech.

Two TTS engines supported:
  * edge   - Microsoft Edge TTS (FREE, no API key, decent quality)
  * openai - OpenAI TTS-1-HD (paid, ~$0.03/1k chars, best quality)

Install dependencies:
    pip install pydub edge-tts openai aiofiles
    # Also install ffmpeg (required by pydub):
    #   macOS  -> brew install ffmpeg
    #   Ubuntu -> sudo apt install ffmpeg
    #   Windows-> https://ffmpeg.org/download.html

Usage:
    # Free run (Edge TTS, no key needed):
    python podcast_tts.py podcast_transcript.txt

    # High quality run (OpenAI):
    OPENAI_API_KEY=sk-... python podcast_tts.py podcast_transcript.txt --engine openai

    # Custom output path:
    python podcast_tts.py transcript.txt --engine openai --output episode_01.mp3
"""

import re
import os
import sys
import asyncio
import argparse
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  FFMPEG CONFIGURATION — Cross-Platform Auto-Detection
# ══════════════════════════════════════════════════════════════════════════════

def _find_ffmpeg() -> Tuple[str, str]:
    """Auto-detect ffmpeg and ffprobe executables across platforms."""
    # Possible executable names
    ffmpeg_exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    ffprobe_exe = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    
    # 1. Check environment variables
    ffmpeg_env = os.environ.get("FFMPEG_PATH", "")
    if ffmpeg_env and os.path.exists(os.path.join(ffmpeg_env, ffmpeg_exe)):
        return os.path.join(ffmpeg_env, ffmpeg_exe), os.path.join(ffmpeg_env, ffprobe_exe)
    
    # 2. Check PATH
    ffmpeg_in_path = shutil.which("ffmpeg")
    ffprobe_in_path = shutil.which("ffprobe")
    if ffmpeg_in_path and ffprobe_in_path:
        return ffmpeg_in_path, ffprobe_in_path
    
    # 3. Check common install locations
    common_paths = []
    if sys.platform == "win32":
        common_paths = [
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
            os.path.expanduser(r"~\ffmpeg\bin"),
        ]
    elif sys.platform == "darwin":
        common_paths = [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
        ]
    else:
        common_paths = [
            "/usr/bin",
            "/usr/local/bin",
            "/snap/bin",
        ]
    
    for path in common_paths:
        ffmpeg_path = os.path.join(path, ffmpeg_exe)
        ffprobe_path = os.path.join(path, ffprobe_exe)
        if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
            return ffmpeg_path, ffprobe_path
    
    # Return None - will fail later with helpful message
    return None, None


FFMPEG_PATH, FFPROBE_PATH = _find_ffmpeg()

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """A single speaker turn in the transcript."""
    index: int
    speaker: str
    text: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSCRIPT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Clean text for TTS: remove artifacts, normalize whitespace, fix common issues."""
    # Remove any remaining markdown
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'__', '', text)
    text = re.sub(r'`', '', text)
    
    # Remove any remaining stage directions
    text = re.sub(r'\s*[\(\[]\s*(?:laughs?|pauses?|sighs?|chuckles?|shrugs?|beat|music|applause)\s*[\)\]]\s*', ' ', text, flags=re.IGNORECASE)
    
    # Remove speaker labels only if they appear at line start (not mid-sentence colons)
    text = re.sub(r'^[A-Z][a-zA-Z]+:\s*', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([.,!?;])', r'\1', text)
    text = re.sub(r'([.,!?;])(?![\s\n])', r'\1 ', text)
    
    return text.strip()


def parse_transcript(filepath: str) -> Tuple[Dict, List[Turn]]:
    """
    Parse the SYNTHETIC MINDS transcript format.
    
    Returns:
        meta  - dict of header key/value pairs (Topic, Model, Turns, Words)
        turns - ordered list of Turn objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    
    # -- Extract metadata lines above the separator --
    meta: Dict[str, str] = {}
    for line in raw.splitlines():
        if re.match(r'^={5,}', line):
            break
        if ':' in line and not line.startswith('='):
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip()
    
    # -- Remove the Research Brief section if present --
    # Research brief starts with "RESEARCH BRIEF" and ends before the next "======" or speaker turn
    raw_transcript = raw
    research_match = re.search(r'RESEARCH BRIEF[\s\S]*?(?=\n={6,}|\n[A-Z][a-zA-Z]+:\s*$)', raw)
    if research_match:
        raw_transcript = raw[:research_match.start()] + raw[research_match.end():]
    
    # -- Split on "Speaker:" labels --
    # Pattern: a name (single or multi-word) on its own line followed by a colon
    # Handles single names like "Arjun:" and multi-word like "Rahul Dravid:"
    # Uses CAPTURING group so speaker names appear in the split result
    speaker_re = re.compile(r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*):\s*$', re.MULTILINE)
    parts = speaker_re.split(raw_transcript)
    
    turns: List[Turn] = []
    idx = 0
    i = 1  # parts[0] is the header preamble
    while i + 1 < len(parts):
        speaker = parts[i].strip().rstrip(':')
        text = _clean_text(parts[i + 1])
        if speaker and text:
            turns.append(Turn(index=idx, speaker=speaker, text=text))
            idx += 1
        i += 2
    
    return meta, turns


# ══════════════════════════════════════════════════════════════════════════════
# 3. VOICE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

# Edge TTS voices - free, no API key
# Indian English voices are prioritized for Indian podcasts
EDGE_VOICE_POOL_INDIA = [
    "en-IN-PrabhatNeural",    # Indian male
    "en-IN-NeerjaNeural",     # Indian female
    "en-IN-KunalNeural",      # Indian male, alternative
    "en-IN-KavyaNeural",      # Indian female, alternative
]

# Fallback voices (non-Indian) if Indian voices not available
EDGE_VOICE_POOL = [
    "en-US-GuyNeural",        # Adult male, natural cadence
    "en-US-JennyNeural",      # Adult female, warm
    "en-US-DavisNeural",      # Male, conversational
    "en-US-AriaNeural",       # Female, expressive
]

# OpenAI TTS voices
OPENAI_VOICE_POOL = [
    "onyx",    # Deep, measured male
    "nova",    # Warm female
    "echo",    # Balanced male
    "shimmer", # Expressive female
    "alloy",   # Neutral
    "fable",   # British male
]


def build_voice_map(turns: List[Turn], engine: str, use_indian_voices: bool = True) -> Dict[str, str]:
    """Auto-assign a unique voice to each speaker found in the transcript."""
    if engine == "openai":
        pool = OPENAI_VOICE_POOL
    elif use_indian_voices:
        pool = EDGE_VOICE_POOL_INDIA + EDGE_VOICE_POOL
    else:
        pool = EDGE_VOICE_POOL
    
    speakers = list(dict.fromkeys(t.speaker for t in turns))  # deduplicated, ordered
    
    voice_map: Dict[str, str] = {}
    for i, speaker in enumerate(speakers):
        voice_map[speaker] = pool[i % len(pool)]
    
    print("\n[Speaker to Voice Mapping]")
    for spk, voice in voice_map.items():
        print(f"  {spk:<20} -> {voice}")
    
    return voice_map


# ══════════════════════════════════════════════════════════════════════════════
# 4. TTS ENGINES
# ══════════════════════════════════════════════════════════════════════════════

# -- 4a. Edge TTS (free) --

async def _edge_speak(text: str, voice: str, path: str) -> None:
    """Async helper: synthesize one segment with Edge TTS."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate="-5%")  # slight slow-down
    await communicate.save(path)


def generate_edge(turns: List[Turn], voice_map: Dict[str, str], out_dir: str) -> List[str]:
    """Generate audio files for every turn using Edge TTS (free, no API key)."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("Install Edge TTS:  pip install edge-tts")
    
    paths: List[str] = []
    print(f"\n[2/3] Generating audio via Edge TTS ({len(turns)} turns)...")
    
    for turn in turns:
        voice = voice_map[turn.speaker]
        path = os.path.join(out_dir, f"seg_{turn.index:04d}_{turn.speaker}.mp3")
        print(f"  [{turn.index+1:>3}/{len(turns)}] {turn.speaker} ({voice}): "
              f"{turn.text[:55]}...")
        asyncio.run(_edge_speak(turn.text, voice, path))
        paths.append(path)
    
    return paths


# -- 4b. OpenAI TTS-1-HD (paid, best quality) --

def generate_openai(turns: List[Turn], voice_map: Dict[str, str], out_dir: str) -> List[str]:
    """Generate audio files for every turn using OpenAI TTS-1-HD."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Install OpenAI SDK:  pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set the OPENAI_API_KEY environment variable before running with --engine openai")
    
    client = OpenAI(api_key=api_key)
    paths: List[str] = []
    print(f"\n[2/3] Generating audio via OpenAI TTS-1-HD ({len(turns)} turns)...")
    
    for turn in turns:
        voice = voice_map[turn.speaker]
        path = os.path.join(out_dir, f"seg_{turn.index:04d}_{turn.speaker}.mp3")
        print(f"  [{turn.index+1:>3}/{len(turns)}] {turn.speaker} ({voice}): "
              f"{turn.text[:55]}...")
        
        response = client.audio.speech.create(
            model="tts-1-hd",       # hd = highest quality
            voice=voice,
            input=turn.text,
            response_format="mp3",
            speed=0.95,             # slightly slower for podcast warmth
        )
        response.stream_to_file(path)
        paths.append(path)
    
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUDIO STITCHING & MASTERING
# ══════════════════════════════════════════════════════════════════════════════

def stitch_and_master(
    seg_paths: List[str],
    turns: List[Turn],
    output_path: str,
    meta: Dict[str, str],
    pause_same_ms: int = 350,
    pause_switch_ms: int = 850,
) -> str:
    """
    Concatenate all segments with speaker-aware pauses, normalize loudness,
    and export as a 192kbps MP3 with podcast ID3 tags.
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
    except ImportError:
        sys.exit(
            "Install pydub + ffmpeg:\n"
            "  pip install pydub\n"
            "  brew install ffmpeg   # macOS\n"
            "  sudo apt install ffmpeg  # Ubuntu"
        )
    
    # Configure ffmpeg paths if auto-detected
    if FFMPEG_PATH and FFPROBE_PATH:
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffprobe = FFPROBE_PATH
    
    print(f"\n[3/3] Stitching {len(seg_paths)} segments...")
    final = AudioSegment.empty()
    
    for i, (path, turn) in enumerate(zip(seg_paths, turns)):
        segment = AudioSegment.from_mp3(path)
        segment = normalize(segment)          # per-segment loudness normalization
        
        # Speaker-aware pause
        if i > 0:
            prev_speaker = turns[i - 1].speaker
            pause_ms = pause_same_ms if prev_speaker == turn.speaker else pause_switch_ms
            final += AudioSegment.silent(duration=pause_ms)
        
        final += segment
    
    # Global loudness normalization (master pass)
    final = normalize(final)
    
    # Export with podcast-grade settings
    topic = meta.get("Topic", "AI & Society")
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    speakers_str = " & ".join(speakers) if len(speakers) <= 2 else ", ".join(speakers)
    
    final.export(
        output_path,
        format="mp3",
        bitrate="192k",
        tags={
            "title": f"Synthetic Minds - {topic}",
            "artist": speakers_str,
            "album": "Synthetic Minds",
            "genre": "Podcast",
            "comment": f"Generated by podcast_tts.py | Model: {meta.get('Model', 'unknown')}",
        },
    )
    
    duration_min = len(final) / 1000 / 60
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  Duration : {duration_min:.1f} min")
    print(f"  File size: {size_mb:.1f} MB")
    
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a multi-speaker podcast transcript to an MP3 audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "transcript",
        help="Path to the transcript .txt file.",
    )
    parser.add_argument(
        "--engine",
        choices=["edge", "openai"],
        default="edge",
        help="TTS engine to use (default: edge - free, no API key required).",
    )
    parser.add_argument(
        "--output",
        default="podcast_output.mp3",
        help="Output MP3 filename (default: podcast_output.mp3).",
    )
    parser.add_argument(
        "--no-indian-voices",
        action="store_true",
        help="Disable Indian-accented voices (use default US voices instead).",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.transcript):
        sys.exit(f"File not found: {args.transcript}")
    
    # -- Check ffmpeg availability --
    if not FFMPEG_PATH or not FFPROBE_PATH:
        print("\n  ERROR: FFmpeg not found!")
        print("  Please install FFmpeg:")
        print("    macOS  -> brew install ffmpeg")
        print("    Ubuntu -> sudo apt install ffmpeg")
        print("    Windows-> download from https://ffmpeg.org/download.html")
        print("  Or set FFMPEG_PATH environment variable to the bin directory.")
        sys.exit(1)
    
    # -- Step 1: Parse transcript --
    print(f"\n[1/3] Parsing transcript: {args.transcript}")
    meta, turns = parse_transcript(args.transcript)
    
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    print(f"  Turns    : {len(turns)}")
    print(f"  Speakers : {', '.join(speakers)}")
    if meta.get("Topic"):
        print(f"  Topic    : {meta['Topic']}")
    
    voice_map = build_voice_map(turns, args.engine, use_indian_voices=not args.no_indian_voices)
    
    # -- Step 2: Generate TTS into temp dir --
    with tempfile.TemporaryDirectory(prefix="podcast_tts_") as tmpdir:
        if args.engine == "openai":
            seg_paths = generate_openai(turns, voice_map, tmpdir)
        else:
            seg_paths = generate_edge(turns, voice_map, tmpdir)
        
        # -- Step 3: Stitch & master --
        output = stitch_and_master(seg_paths, turns, args.output, meta)
    
    print(f"\n  Podcast ready: {output}")
    print(f"    Speakers : {', '.join(speakers)}")
    print(f"    Engine   : {args.engine}")


if __name__ == "__main__":
    main()
