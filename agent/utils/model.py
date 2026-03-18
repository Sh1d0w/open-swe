import os

from langchain.chat_models import init_chat_model

OPENAI_RESPONSES_WS_BASE_URL = "wss://api.openai.com/v1"
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "sk-placeholder")


def make_model(model_id: str, **kwargs: dict):
    model_kwargs = kwargs.copy()

    if model_id.startswith("openai:"):
        model_kwargs["base_url"] = OPENAI_RESPONSES_WS_BASE_URL
        model_kwargs["use_responses_api"] = True

    elif model_id.startswith("lmstudio:"):
        # Extract model name after prefix (e.g., "llama3" from "lmstudio:llama3")
        model_name = model_id[len("lmstudio:"):]
        model_kwargs["model"] = f"openai:{model_name}"
        model_kwargs["base_url"] = LMSTUDIO_BASE_URL
        model_kwargs["api_key"] = LMSTUDIO_API_KEY

    return init_chat_model(model=model_id, **model_kwargs)
