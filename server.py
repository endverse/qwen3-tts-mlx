"""FastAPI server for programmatic Qwen3-TTS access."""

import argparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from tts import (
    VOICES,
    ALL_MODELS,
    PRESET_MODELS,
    DESIGN_MODELS,
    CLONE_MODELS,
    LANGUAGES,
    get_model_status,
    load_saved_voices,
    generate_preset_audio,
    generate_preset_audio_stream,
    generate_design_audio,
    generate_design_audio_stream,
    generate_clone_audio,
)

app = FastAPI(title="Qwen3-TTS API", version="1.0.0")


# --- Request models ---

class PresetRequest(BaseModel):
    text: str
    voice: str = VOICES[0]
    instruct: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    model: str = list(PRESET_MODELS.keys())[0]
    streaming_interval: float = Field(default=2.0, ge=0.5, le=10.0)


class DesignRequest(BaseModel):
    text: str
    instruct: str
    language: str = "Auto"
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    streaming_interval: float = Field(default=2.0, ge=0.5, le=10.0)


class CloneRequest(BaseModel):
    text: str
    voice: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    model: str = list(CLONE_MODELS.keys())[0]


# --- Endpoints ---

@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.get("/v1/voices")
def list_voices():
    preset = [v.split(" (")[0] for v in VOICES]
    saved = list(load_saved_voices().keys())
    return {"preset": preset, "saved": saved}


@app.get("/v1/models")
def list_models():
    return {"models": get_model_status()}


@app.post("/v1/tts/generate")
def generate_preset(req: PresetRequest):
    try:
        _audio, metadata = generate_preset_audio(
            text=req.text,
            voice=req.voice,
            instruct=req.instruct,
            temperature=req.temperature,
            model_name=req.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return FileResponse(
        metadata["path"],
        media_type="audio/wav",
        filename=metadata["filename"],
    )


@app.post("/v1/tts/design")
def generate_design(req: DesignRequest):
    try:
        _audio, metadata = generate_design_audio(
            text=req.text,
            instruct=req.instruct,
            language=req.language,
            temperature=req.temperature,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return FileResponse(
        metadata["path"],
        media_type="audio/wav",
        filename=metadata["filename"],
    )


@app.post("/v1/tts/clone")
def generate_clone(req: CloneRequest):
    try:
        _audio, metadata = generate_clone_audio(
            text=req.text,
            saved_voice=req.voice,
            temperature=req.temperature,
            model_name=req.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return FileResponse(
        metadata["path"],
        media_type="audio/wav",
        filename=metadata["filename"],
    )


@app.post("/v1/tts/generate/stream")
def stream_preset(req: PresetRequest):
    import numpy as np

    def audio_chunks():
        for chunk in generate_preset_audio_stream(
            text=req.text,
            voice=req.voice,
            instruct=req.instruct,
            temperature=req.temperature,
            model_name=req.model,
            streaming_interval=req.streaming_interval,
        ):
            yield chunk.astype(np.float32).tobytes()

    try:
        gen = audio_chunks()
        first = next(gen)
    except StopIteration:
        raise HTTPException(status_code=503, detail="Generation failed - no audio produced")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from itertools import chain
    return StreamingResponse(
        chain([first], gen),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000", "X-Sample-Format": "float32"},
    )


@app.post("/v1/tts/design/stream")
def stream_design(req: DesignRequest):
    import numpy as np

    def audio_chunks():
        for chunk in generate_design_audio_stream(
            text=req.text,
            instruct=req.instruct,
            language=req.language,
            temperature=req.temperature,
            streaming_interval=req.streaming_interval,
        ):
            yield chunk.astype(np.float32).tobytes()

    try:
        gen = audio_chunks()
        first = next(gen)
    except StopIteration:
        raise HTTPException(status_code=503, detail="Generation failed - no audio produced")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from itertools import chain
    return StreamingResponse(
        chain([first], gen),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000", "X-Sample-Format": "float32"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
