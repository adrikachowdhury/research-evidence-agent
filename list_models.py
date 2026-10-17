r"""
List available Gemini API models that support text generation
"""

from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name, "-", model.display_name)