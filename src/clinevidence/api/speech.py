"""Speech transcription and synthesis endpoints."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
)

from clinevidence.dependencies import get_app_settings
from clinevidence.models.requests import SpeechRequest
from clinevidence.models.responses import (
    SpeechResponse,
    TranscribeResponse,
)
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Speech"])

_ALLOWED_AUDIO_FORMATS = frozenset(
    {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}
)


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio to text",
    description=(
        "Accepts an audio file and returns the transcribed text "
        "using the Eleven Labs API."
    ),
)
async def transcribe_audio(
    file: UploadFile,
    settings: Settings = Depends(get_app_settings),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file to text."""
    filename = file.filename or "audio.mp3"
    ext = Path(filename).suffix.lstrip(".").lower()

    if ext not in _ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Audio format '.{ext}' not supported. "
                f"Allowed: {sorted(_ALLOWED_AUDIO_FORMATS)}"
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(
            api_key=(settings.elevenlabs_api_key.get_secret_value())
        )
        transcription = client.speech_to_text.convert(
            audio=content,
            model_id="scribe_v1",
        )
        text: str = str(getattr(transcription, "text", transcription)).strip()
        logger.info(
            "Audio transcribed",
            extra={
                "text_len": len(text),
                "audio_size_kb": len(content) // 1024,
            },
        )
        return TranscribeResponse(text=text)

    except Exception as exc:
        logger.error("Audio transcription failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("Transcription service unavailable. Please try again."),
        ) from exc


@router.post(
    "/synthesize",
    response_model=SpeechResponse,
    summary="Synthesize speech from text",
    description=(
        "Converts text to speech using Eleven Labs "
        "and returns a URL to the generated audio file."
    ),
)
async def synthesize_speech(
    request: SpeechRequest,
    settings: Settings = Depends(get_app_settings),
) -> SpeechResponse:
    """Convert text to speech and return the audio file URL."""
    voice_id = request.voice_id or settings.elevenlabs_voice_id

    try:
        from elevenlabs import VoiceSettings
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(
            api_key=(settings.elevenlabs_api_key.get_secret_value())
        )
        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=request.text,
            model_id=settings.elevenlabs_model_id,
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
            ),
        )

        audio_bytes = b"".join(audio_stream)

        audio_dir = Path(settings.upload_dir) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = audio_dir / audio_filename
        audio_path.write_bytes(audio_bytes)

        audio_url = f"/uploads/audio/{audio_filename}"

        logger.info(
            "Speech synthesised",
            extra={
                "text_len": len(request.text),
                "audio_kb": len(audio_bytes) // 1024,
                "filename": audio_filename,
            },
        )
        return SpeechResponse(audio_url=audio_url)

    except Exception as exc:
        logger.error("Speech synthesis failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("Speech synthesis service unavailable. Please try again."),
        ) from exc
