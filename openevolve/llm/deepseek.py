"""
DeepSeek API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class DeepSeekLLM(LLMInterface):
    """LLM interface using DeepSeek-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name or "deepseek-chat"
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base or "https://api.deepseek.com"
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, 'random_seed', None)

        # Set up API client (DeepSeek is OpenAI-compatible)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        logger.info(f"Initialized DeepSeek LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add seed parameter for reproducibility if configured
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content 