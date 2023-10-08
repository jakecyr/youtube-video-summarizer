"""Module for interacting with the OpenAI API."""

import os
from enum import Enum
from typing import Any, TypedDict

import openai
from loguru import logger


class TextGenerationError(BaseException):
    """Exception raised for text generation errors."""


class ChatRole(str, Enum):
    """Enum representing the role options for a ChatCompletion input message."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class ChatCompletionMessage(TypedDict):
    """An input message to the ChatCompletion API endpoint.

    Args:
        role: The role of the message.
        content: The content of the message.
    """

    role: ChatRole
    content: str


class MessageRole(str, Enum):
    """Enum representing the role options for a ChatCompletion input message.

    Args:
        USER: The user role.
        ASSISTANT: The assistant role.
        SYSTEM: The system role.
    """

    __slots__ = ()
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class OpenAIClient:
    """Client for interacting with the OpenAI API.

    Attributes
        api_key: The OpenAI API key.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAIClient instance.

        Args:
            api_key: The OpenAI API key.
        """
        if not api_key and "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "Expected api_key parameter or OPENAI_API_KEY env var to be set.",
            )

        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def generate_chat_completion(
        self,
        user_prompt: str,
        *,
        model: str = "gpt-3.5-turbo-0613",
        system_prompt: str | None = None,
        temperature: float = 0.5,
    ) -> dict:
        """Generate a chat completion.

        Args:
            user_prompt: The user prompt to generate the chat completion from.
            model: The model to use for the API.
            temperature: The temperature to use for the model.

        Returns:
            The response from the model.
        """
        logger.debug("Generating chat completion...")

        messages: list[ChatCompletionMessage] = [
            ChatCompletionMessage(role=ChatRole.USER, content=user_prompt),
        ]

        if system_prompt:
            messages.insert(
                0, ChatCompletionMessage(role=ChatRole.SYSTEM, content=system_prompt)
            )

        response: Any = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        message_dict: dict = response["choices"][0]["message"]
        return message_dict

    async def generate_chat_completion_async(
        self,
        user_prompt: str,
        *,
        model: str = "gpt-3.5-turbo-0613",
        system_prompt: str | None = None,
        temperature: float = 0.5,
    ) -> dict:
        """Generate a chat completion.

        Args:
            user_prompt: The user prompt to generate the chat completion from.
            model: The model to use for the API.
            temperature: The temperature to use for the model.

        Returns:
            The response from the model.
        """
        logger.debug("Generating async chat completion...")

        messages: list[ChatCompletionMessage] = [
            ChatCompletionMessage(role=ChatRole.USER, content=user_prompt),
        ]

        if system_prompt:
            messages.insert(
                0, ChatCompletionMessage(role=ChatRole.SYSTEM, content=system_prompt)
            )

        response: Any = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        message_dict: dict = response["choices"][0]["message"]
        return message_dict
