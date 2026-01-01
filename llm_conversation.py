import openai
from typing import List, Dict, Optional, Union, Any
import os
import base64
from dotenv import load_dotenv
from configuration import llm_config


class LLMConversation:
    """
    A class for handling conversations with OpenAI's LLM models, including Vision API support for image inputs.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.7):
        """
        Initialize the LLM conversation handler.
        Args:
            api_key: API key. If None, will use config.llm_api_key
            base_url: Base URL for the API. If None, will use config.llm_base_url
            model: The model to use. If None, will use config.llm_model
            temperature: Controls randomness in the response (0.0 to 1.0)
        """
        self.api_key = api_key or llm_config.llm_api_key
        self.base_url = base_url or llm_config.llm_base_url
        self.model = model or llm_config.llm_model
        self.temperature = temperature
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def send_message(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            # Extract the response content
            assistant_message = response.choices[0].message.content
            if assistant_message.startswith("```json"):
                assistant_message = assistant_message[7:]
            if assistant_message.endswith("```"):
                assistant_message = assistant_message[:-3]
            return assistant_message
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error sending message: {str(e)}")

    def set_model(self, model: str):
        """
        Change the model being used.

        Args:
            model: The new model name
        """
        self.model = model

    def set_temperature(self, temperature: float):
        """
        Change the temperature setting.

        Args:
            temperature: New temperature value (0.0 to 1.0)
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature

    def encode_image_to_base64(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> str:
        """
        Encode an image to base64 format for use with Vision API.

        Args:
            image_path: Path to the image file
            image_bytes: Raw image bytes

        Returns:
            Base64 encoded image string
        """
        if image_path:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
        elif image_bytes is None:
            raise ValueError("Either image_path or image_bytes must be provided")

        # Encode to base64
        encoded_string = base64.b64encode(image_bytes).decode('utf-8')
        return encoded_string

    def create_message_with_image(self, text: str, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None, image_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a message containing both text and image for Vision API.

        Args:
            text: The text content of the message
            image_path: Path to the image file
            image_bytes: Raw image bytes
            image_url: URL of the image (if already hosted)

        Returns:
            Message dictionary compatible with Vision API
        """
        content = [{"type": "text", "text": text}]

        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_path or image_bytes:
            base64_image = self.encode_image_to_base64(image_path, image_bytes)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        return {
            "role": "user",
            "content": content
        }

    def send_message_with_image(self, messages: List[Dict[str, Any]], text: str, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None, image_url: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Send a message with both text and image to the LLM.

        Args:
            text: The text content
            image_path: Path to the image file
            image_bytes: Raw image bytes
            image_url: URL of the image
            max_tokens: Maximum tokens in response

        Returns:
            Assistant's response
        """
        messages.append(self.create_message_with_image(text, image_path, image_bytes, image_url))
        return self.send_message(messages, max_tokens)

if __name__ == "__main__":
    load_dotenv()
    llm_conversation = LLMConversation()
    response = llm_conversation.send_message_with_image([], "解释图片中的内容", image_path=r"C:\Users\29853\Desktop\PolyQuery\public\屏幕截图.png")
    print(response)
