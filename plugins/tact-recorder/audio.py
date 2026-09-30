"""Recording -> transcript: ffmpeg splits the audio into STT-sized chunks, each goes through the
configured STT provider (``tools.transcription_tools.transcribe_audio``), transcripts join in order.

Chunks are 16 kHz mono AAC (the same encode profile the STT pipeline uses) of ``CHUNK_SECONDS``
each: ~2.4 MB per 10 minutes, far below the 25 MB cloud upload cap. Without ffmpeg the original
file is sent as one piece when it fits under the cap.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 600
FFMPEG_TIMEOUT = 1800


class TranscriptionError(Exception):
    """User-facing reason the recording could not be transcribed."""


def split_audio(path: Path, work_dir: Path) -> List[Path]:
    """Chunk files in playback order."""
    from tools.transcription_audio import _STT_M4A_ENCODE_ARGS, _find_ffmpeg_binary, _run_quiet
    from tools.transcription_common import MAX_FILE_SIZE

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        if path.stat().st_size > MAX_FILE_SIZE:
            raise TranscriptionError("ffmpeg is not available to split this long recording.")
        return [path]
    # -movflags does not apply per segment; everything else is the shared STT encode profile.
    encode = [a for a in _STT_M4A_ENCODE_ARGS if a not in ("-movflags", "+faststart")]
    try:
        _run_quiet([ffmpeg, "-y", "-i", str(path), *encode, "-f", "segment",
                    "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
                    str(work_dir / "chunk_%04d.m4a")], timeout=FFMPEG_TIMEOUT)
    except Exception as exc:
        logger.warning("tact-recorder: ffmpeg could not split the recording: %s", type(exc).__name__)
        raise TranscriptionError("The audio could not be read. Please send it as m4a, mp3 or ogg.") from exc
    chunks = sorted(work_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise TranscriptionError("The recording has no audio track.")
    if any(c.stat().st_size > MAX_FILE_SIZE for c in chunks):
        raise TranscriptionError("A part of the recording is too large for the transcription service.")
    return chunks


def transcribe_chunks(chunks: List[Path]) -> str:
    from tools import transcription_tools

    parts = []
    for i, chunk in enumerate(chunks, 1):
        result = transcription_tools.transcribe_audio(str(chunk), None, "tact-recorder")
        if not result.get("success"):
            # The provider's error names the failure (quota, key, format); it never carries audio content.
            logger.warning("tact-recorder: transcription of part %d/%d failed: %s",
                           i, len(chunks), result.get("error", "unknown error"))
            raise TranscriptionError(f"Transcription failed on part {i} of {len(chunks)}.")
        text = (result.get("transcript") or "").strip()
        if text:
            parts.append(text)
    transcript = "\n".join(parts).strip()
    if not transcript:
        raise TranscriptionError("No speech was found in the recording.")
    return transcript


def transcribe_recording(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="tact-recorder-") as tmp:
        chunks = split_audio(path, Path(tmp))
        logger.info("tact-recorder: transcribing %d part(s)", len(chunks))
        return transcribe_chunks(chunks)
