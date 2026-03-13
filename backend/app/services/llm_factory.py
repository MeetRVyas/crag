"""
Centralised factory for LLM and Embedding objects.

Supported LLM providers
-----------------------
    ollama            → ChatOllama  (local / docker Ollama server)
    groq              → ChatGroq    (Groq Cloud API)
    anthropic         → ChatAnthropic (Claude API)
    huggingface_api   → ChatHuggingFace via HuggingFaceEndpoint (remote HF Inference API)
    huggingface_local → ChatHuggingFace via HuggingFacePipeline (runs locally on CPU/GPU)
    google            → ChatGoogleGenerativeAI (Gemini API)

Supported Embedding providers
------------------------------
    ollama            → OllamaEmbeddings
    huggingface       → HuggingFaceEmbeddings  (sentence-transformers, local)
    google            → GoogleGenerativeAIEmbeddings

HuggingFace API would be costly and unnecessary for Embeddings purpose

Auto-fallback
-------------
    groq + anthropic have no embedding API.
    When either is used as the embedding provider it falls back automatically
    to HuggingFace local (sentence-transformers/all-MiniLM-L6-v2).
"""

import os
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import settings


# Default model used when the huggingface embedding provider is selected
# but no model name is given
_HF_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_OLLAMA_DEFAULT_EMBEDDING_MODEL = "embeddinggemma:300m"


# ===========================================================================
# LLM factory
# ===========================================================================

def _llm_ollama(model: str, api_keys: dict) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    from app.services.ollama_service import get_ollama_service

    return ChatOllama(
        model = get_ollama_service().validate_and_ensure_llm(model),
        temperature = 0,
            base_url = settings.OLLAMA_BASE_URL
    )


def _llm_groq(model: str, api_keys: dict) -> BaseChatModel:
    from langchain_groq import ChatGroq

    api_key = api_keys.get("groq") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Groq API key is required. "
            "Provide it via api_keys['groq'] or the GROQ_API_KEY env var."
        )
    return ChatGroq(
            model = model,
            temperature = 0,
            api_key = api_key,
        )


def _llm_anthropic(model: str, api_keys: dict) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Anthropic API key is required. "
            "Provide it via api_keys['anthropic'] or the ANTHROPIC_API_KEY env var."
        )
    return ChatAnthropic(
            model = model,
            temperature = 0,
            api_key = api_key,
        )


def _llm_google(model: str, api_keys: dict) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Google API key is required. "
            "Provide it via api_keys['google'] or the GOOGLE_API_KEY env var."
        )
    return ChatGoogleGenerativeAI(
            model = model,
            temperature = 0,
            google_api_key = api_key,
        )


def _llm_huggingface_api(model: str, api_keys: dict) -> BaseChatModel:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    # Key is stored under "huggingface" regardless of whether the provider
    # string is "huggingface_api" or "huggingface_local"
    api_key = api_keys.get("huggingface") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not api_key:
        raise ValueError(
            "HuggingFace API token is required. "
            "Provide it via api_keys['huggingface'] or HUGGINGFACEHUB_API_TOKEN."
        )
    endpoint = HuggingFaceEndpoint(
        repo_id = model,
        task = "text-generation",
        temperature = 0.01, # HF endpoint rejects exactly 0.0
        max_new_tokens = 1024,
        huggingfacehub_api_token = api_key,
    )
    return ChatHuggingFace(llm = endpoint)


def _llm_huggingface_local(model: str, api_keys: dict) -> BaseChatModel:
    # No API key needed — model runs locally via transformers
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    pipeline = HuggingFacePipeline.from_model_id(
        model_id = model,
        task = "text-generation",
        pipeline_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.01, # transformers rejects exactly 0
            "do_sample": True,
        },
        device = -1,  # -1 = CPU; set to 0 for first GPU
    )
    return ChatHuggingFace(llm = pipeline)


# ---------------------------------------------------------------------------
# LLM registry  —  provider string → builder function
# ---------------------------------------------------------------------------

_LLM_REGISTRY: dict[str, Callable[[str, dict], BaseChatModel]] = {
    "ollama":            _llm_ollama,
    "groq":              _llm_groq,
    "anthropic":         _llm_anthropic,
    "google":            _llm_google,
    "huggingface_api":   _llm_huggingface_api,
    "huggingface_local": _llm_huggingface_local,
}


# ===========================================================================
# Embeddings factory
# ===========================================================================


def _emb_ollama(model: str, api_keys: dict) -> Embeddings:
    from langchain_ollama.embeddings import OllamaEmbeddings
    from app.services.ollama_service import get_ollama_service

    return OllamaEmbeddings(
        model = get_ollama_service().validate_and_ensure_embedding(model),
        base_url = settings.OLLAMA_BASE_URL
    )


def _emb_huggingface(model: str, api_keys: dict) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model or _HF_DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _emb_google(model: str, api_keys: dict) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Google API key is required for Google embeddings. "
            "Provide it via api_keys['google'] or GOOGLE_API_KEY."
        )
    return GoogleGenerativeAIEmbeddings(
            model = model,
            google_api_key = api_key,
        )


def _emb_ollama_fallback(model: str, api_keys: dict) -> Embeddings:
    """
    Fallback used when the LLM provider has no embedding API (groq, anthropic).
    Always uses settings.EMBEDDING_MODEL, ignoring the model argument.
    """
    from app.config import settings

    if settings.EMBEDDING_PROVIDER == "ollama" :
        fallback_model = settings.EMBEDDING_MODEL
    else :
        fallback_model = _OLLAMA_DEFAULT_EMBEDDING_MODEL
    print(
        "Provider has no embedding API — falling back to Ollama: %s", fallback_model
    )
    return _emb_ollama(
        model = fallback_model,
        api_keys = {}
        )


# ---------------------------------------------------------------------------
# Embedding registry  —  provider string → builder function
# ---------------------------------------------------------------------------

_EMBEDDING_REGISTRY: dict[str, Callable[[str, dict], Embeddings]] = {
    "ollama":      _emb_ollama,
    "huggingface": _emb_huggingface,
    "google":      _emb_google,
    # Providers with no embedding API → Ollama fallback
    "groq":        _emb_ollama_fallback,
    "anthropic":   _emb_ollama_fallback,
}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(
    provider: str,
    model: str,
    api_keys: Optional[dict] = None,
) -> BaseChatModel:
    """Build and return a chat-capable LLM for the selected provider."""
    api_keys = api_keys or {}
    provider = provider.lower()

    builder = _LLM_REGISTRY.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose one of: {', '.join(sorted(_LLM_REGISTRY))}"
        )
    
    return builder(model, api_keys)


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def build_embeddings(
    provider: str,
    model: str,
    api_keys: Optional[dict] = None,
) -> Embeddings:
    """Build and return an Embeddings object for the selected provider."""
    api_keys = api_keys or {}
    provider = provider.lower()

    builder = _EMBEDDING_REGISTRY.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Choose one of: {', '.join(sorted(_EMBEDDING_REGISTRY))}"
        )

    return builder(model, api_keys)