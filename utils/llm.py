import os
from openai import OpenAI, AsyncOpenAI


def create_chat_completion(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
):
    client = OpenAI(
        api_key="sk--",
        base_url=os.environ.get("TRAFFICR1_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        **kwargs,
    )
    return response
