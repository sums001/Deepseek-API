"""A frozen catalog used by the standalone dropdown verification.

The shapes here are copied from the structure the live service actually
returns (``supported_endpoint_types`` and ``architecture.input_modalities``),
then extended with the record kinds the offline suite has no live example of:
an embedding model, an image-generation model, a video model, and a rerank
model. Nothing about a model's capability is inferred from its name.
"""

from __future__ import annotations

CATALOG_FIXTURE = {
    "object": "list",
    "data": [
        {
            "id": "vendor/text-only",
            "object": "model",
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text"]},
            "context_length": 128000,
        },
        {
            "id": "vendor/vision-chat",
            "object": "model",
            "supported_endpoint_types": ["openai", "anthropic"],
            "architecture": {"input_modalities": ["text", "image"]},
            "context_length": 200000,
        },
        {
            "id": "vendor/reasoning-text",
            "object": "model",
            "supported_endpoint_types": ["openai-response"],
            "architecture": {"input_modalities": ["text"]},
            "reasoning_efforts": ["low", "medium", "high", "xhigh"],
        },
        {
            "id": "vendor/embedding",
            "object": "model",
            "supported_endpoint_types": ["embeddings"],
        },
        {
            "id": "vendor/image-gen",
            "object": "model",
            "supported_endpoint_types": ["image-generation"],
        },
        {
            "id": "vendor/video-gen",
            "object": "model",
            "supported_endpoint_types": ["openai-video"],
        },
        {
            "id": "vendor/reranker",
            "object": "model",
            "supported_endpoint_types": ["jina-rerank"],
        },
    ],
}

# What each capability must end up with, if the filtering is correct.
EXPECTED = {
    "chat": ["vendor/text-only", "vendor/vision-chat", "vendor/reasoning-text"],
    "multimodal_image": ["vendor/vision-chat"],
    "embedding": ["vendor/embedding"],
    "image": ["vendor/image-gen"],
    "video": ["vendor/video-gen"],
    "rerank": ["vendor/reranker"],
}
