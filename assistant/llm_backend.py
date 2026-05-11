#!/usr/bin/env python3
"""Local Ollama backend plus cache helpers for the assistant layer.

This module is deliberately small. It does not know anything about runs,
journals, or prompt assembly; it only knows how to:
1. read runtime config from env vars,
2. compute a stable prompt hash,
3. read/write cached responses,
4. send one prompt pair to Ollama and return plain text.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class LLMConfig:
    model: str
    base_url: str
    temperature: float | None
    timeout_sec: int


def load_llm_config() -> LLMConfig:
    """Read Ollama runtime configuration from env vars."""
    provider = os.environ.get("ASSISTANT_LLM_PROVIDER", "ollama").strip()
    if provider != "ollama":
        raise RuntimeError(f"Unsupported ASSISTANT_LLM_PROVIDER: {provider}. Only `ollama` is supported.")
    model = os.environ.get("ASSISTANT_LLM_MODEL", "").strip()
    default_base_url = "http://127.0.0.1:11434/api/generate"
    base_url = os.environ.get("ASSISTANT_LLM_BASE_URL", default_base_url).strip()
    temperature_raw = os.environ.get("ASSISTANT_LLM_TEMPERATURE", "0.2").strip()
    temperature = None if temperature_raw == "" else float(temperature_raw)
    timeout_sec = int(os.environ.get("ASSISTANT_LLM_TIMEOUT_SEC", "180"))
    return LLMConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout_sec=timeout_sec,
    )


def build_prompt_hash(system_prompt: str, user_prompt: str, *, model: str) -> str:
    """Compute a stable cache key for one prompt pair and one model tag."""
    payload = {
        "provider": "ollama",
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cache_path_for(cache_dir: Path, experiment_signature: str) -> Path:
    """Map one experiment signature to its cache file path."""
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_signature)
    return cache_dir / f"{safe_name}.json"


def load_cached_response(
    cache_dir: Path,
    *,
    experiment_signature: str,
    prompt_hash: str,
) -> dict | None:
    """Load a cached response only if its prompt hash still matches."""
    path = cache_path_for(cache_dir, experiment_signature)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("prompt_hash") != prompt_hash:
        return None
    return payload


def save_cached_response(
    cache_dir: Path,
    *,
    experiment_signature: str,
    prompt_hash: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
) -> Path:
    """Persist one Ollama response for later reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(cache_dir, experiment_signature)
    payload = {
        "experiment_signature": experiment_signature,
        "prompt_hash": prompt_hash,
        "provider": "ollama",
        "model": model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_text": response_text,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def call_ollama_api(config: LLMConfig, *, system_prompt: str, user_prompt: str) -> str:
    """Send one prompt pair to the local Ollama HTTP API and return plain text."""
    if not config.model:
        raise RuntimeError("ASSISTANT_LLM_MODEL is not set.")
    if not config.base_url:
        raise RuntimeError("ASSISTANT_LLM_BASE_URL is not set for provider=ollama.")

    body = {
        "model": config.model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
    }
    if config.temperature is not None:
        body["options"] = {"temperature": config.temperature}

    request = urllib.request.Request(
        config.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "invalid model name" in details.lower():
            details = (
                f"{details}\nHint: run `ollama list` and set ASSISTANT_LLM_MODEL "
                "to an exact installed tag, for example `qwen2.5:1.5b`."
            )
        raise RuntimeError(f"LLM HTTP error {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM network error: {exc}") from exc

    text = payload.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama response did not contain text in `response`.")
    return text.strip()


def call_llm(*, system_prompt: str, user_prompt: str) -> tuple[LLMConfig, str]:
    """Thin wrapper that loads config and performs one Ollama call."""
    config = load_llm_config()
    return config, call_ollama_api(config, system_prompt=system_prompt, user_prompt=user_prompt)
