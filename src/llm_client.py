# Credit Risk XAI - LLM Client
# ==============================
# Isolated wrapper around the Groq API. Swap the provider/model here
# without touching generate_explanations.py or any other module.
#
# Uses: Groq Cloud + LLaMA 3.3 70B (free tier)
# Auth: GROQ_API_KEY from .env file

import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError, NotFoundError

# Load environment variables from .env file
load_dotenv()

# ---- Configuration ---------------------------------------------------------
MODEL_NAME  = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY = 10   # seconds between retries (Groq free tier: ~30 req/min)


def generate_explanation(prompt: str, temperature: float = 0.2) -> str:
    """
    Send a prompt to Groq's LLaMA 3.3 70B and return the text response.

    Parameters
    ----------
    prompt : str
        The fully formatted prompt to send.
    temperature : float
        Controls randomness. 0.0 = deterministic, 1.0 = very creative.
        We default to 0.2 for consistent, factual explanations.

    Returns
    -------
    str
        The model's text response.

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in the project root "
            "with: GROQ_API_KEY=your_key_here"
        )

    client = Groq(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clear, empathetic financial advisor "
                            "who explains automated credit decisions in plain "
                            "language. Avoid jargon. Be concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt < MAX_RETRIES:
                print(f"    [RATE LIMIT] Attempt {attempt}/{MAX_RETRIES} - "
                      f"waiting {RETRY_DELAY}s before retry ...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Groq rate limit hit {MAX_RETRIES} times. "
                    "Wait a minute and try again, or upgrade your plan."
                )

        except NotFoundError as e:
            # Model doesn't exist — no point retrying
            raise RuntimeError(
                f"Model '{MODEL_NAME}' not found on Groq. "
                f"Update MODEL_NAME in llm_client.py. Error: {e}"
            )

        except APIStatusError as e:
            if attempt < MAX_RETRIES:
                print(f"    [API ERROR] {e} - retrying in {RETRY_DELAY}s ...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Groq API error after {MAX_RETRIES} attempts: {e}")

