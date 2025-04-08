import os
import logging
from typing import List, Dict, Any
from openai import OpenAI, APIError, RateLimitError
from dotenv import load_dotenv

class ModelManager:
    """Handles OpenAI client initialization, model selection, and LLM calls."""

    DEFAULT_MODELS = {
        "planning": "gpt-4o-mini",
        "code_gen": "gpt-4o-mini",
        "correction": "gpt-4o-mini",
        "summarization": "gpt-3.5-turbo", # Use a faster/cheaper model for summarization
        "default": "gpt-4o-mini"
    }

    def __init__(self, model_config: Dict[str, str] | None = None):
        """Initializes the ModelManager and the OpenAI client.

        Args:
            model_config (Dict[str, str] | None, optional):
                A dictionary mapping task types ('planning', 'code_gen', etc.)
                to specific model names. Defaults to DEFAULT_MODELS.
        """
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.client = OpenAI()
        self.models = self.DEFAULT_MODELS.copy()
        if model_config:
            self.models.update(model_config)
        logging.info(f"ModelManager initialized with models: {self.models}")

    def get_model_for_task(self, task_type: str) -> str:
        """Returns the appropriate model name for a given task type."""
        return self.models.get(task_type, self.models["default"])

    def call_llm(
        self,
        task_type: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calls the appropriate OpenAI LLM model for the given task type.

        Args:
            task_type (str): The type of task (e.g., 'planning', 'code_gen').
            messages (List[Dict[str, str]]): The message list for the chat completion.
            **kwargs: Additional arguments to pass to `client.chat.completions.create`
                      (e.g., temperature, max_tokens, response_format).

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "success" (bool): Whether the API call was successful.
                - "content" (str | None): The response content if successful, None otherwise.
                - "error" (str | None): An error message if unsuccessful, None otherwise.
        """
        model_name = self.get_model_for_task(task_type)
        logging.info(f"Calling LLM (Model: {model_name}, Task: {task_type}) with {len(messages)} messages.")

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs
            )
            content = response.choices[0].message.content
            logging.info(f"LLM call successful (Task: {task_type}). Response length: {len(content) if content else 0}")
            return {
                "success": True,
                "content": content.strip() if content else "",
                "error": None
            }
        except (APIError, RateLimitError) as e:
            logging.error(f"OpenAI API error during LLM call (Task: {task_type}, Model: {model_name}): {e}", exc_info=True)
            return {
                "success": False,
                "content": None,
                "error": f"OpenAI API Error: {type(e).__name__} - {e}"
            }
        except Exception as e:
            logging.error(f"Unexpected error during LLM call (Task: {task_type}, Model: {model_name}): {e}", exc_info=True)
            return {
                "success": False,
                "content": None,
                "error": f"Unexpected Error: {type(e).__name__} - {e}"
            } 