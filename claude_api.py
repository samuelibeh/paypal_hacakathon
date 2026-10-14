"""
Claude API integration for PayPal Hackathon project.
Supports both Anthropic API and AWS Bedrock.
"""

import os
import json
import urllib.parse
import subprocess
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ClaudeAPI:
    """Wrapper for Claude API interactions."""

    def __init__(self, model: str = "claude-opus-5", use_bedrock: bool = False):
        """
        Initialize Claude API client.
        
        Args:
            model: Claude model to use (default: claude-opus-5)
            use_bedrock: Use AWS Bedrock instead of Anthropic API (default: False)
        """
        self.use_bedrock = use_bedrock
        self.model = model
        self.conversation_history = []
        
        if use_bedrock:
            self.bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            if not self.bedrock_api_key:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK not found in environment. "
                    "Please set it in .env file or as environment variable."
                )
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Please set it in .env file or as environment variable."
                )
            
            workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
            
            if workspace_id:
                self.client = Anthropic(
                    api_key=api_key,
                    default_headers={"anthropic-workspace-id": workspace_id}
                )
            else:
                self.client = Anthropic(api_key=api_key)

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Claude's response text
        """
        if self.use_bedrock:
            response_text = self._bedrock_chat(user_message)
        else:
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=self.conversation_history
            )
            response_text = response.content[0].text
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
        
        return response_text

    def _bedrock_chat(self, user_message: str) -> str:
        """Call AWS Bedrock API for chat."""
        # Build messages in Bedrock format
        messages = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append({
                    "role": "user",
                    "content": [{"text": msg["content"]}]
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"text": msg["content"]}]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })
        
        body = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 1024
            },
            "additionalModelRequestFields": {
                "top_k": 250
            }
        }
        
        # URL encode the model ARN
        model_arn = "arn:aws:bedrock:us-east-1:022677638266:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"
        encoded_model = urllib.parse.quote(model_arn, safe='')
        
        url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{encoded_model}/converse"
        
        # Use curl for the request
        curl_cmd = [
            "curl", "-sS",
            "-X", "POST",
            url,
            "-H", f"Authorization: Bearer {self.bedrock_api_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(body)
        ]
        
        try:
            result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
            response_data = json.loads(result.stdout)
            response_text = response_data["output"]["message"]["content"][0]["text"]
            
            # Update conversation history with the response
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            return response_text
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON response from Bedrock: {result.stdout}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Bedrock API error: {e.stderr}")

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Claude API - Testing both providers")
    print("=" * 60)
    
    # Test Anthropic API
    print("\n[1] Testing Anthropic Claude API...")
    try:
        claude = ClaudeAPI()
        print(f"✓ Initialized: {claude.model}")
        response = claude.chat("What is 2 + 2?")
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}")
        if "credit" in str(e).lower():
            print("  → Need to add credits at console.anthropic.com/account/billing")
        else:
            print(f"  → {e}")
    
    # Test Bedrock API
    print("\n[2] Testing AWS Bedrock Claude API...")
    try:
        bedrock = ClaudeAPI(use_bedrock=True)
        print(f"✓ Initialized: Bedrock (Haiku 4.5)")
        response = bedrock.chat("What is 2 + 2?")
        print(f"✓ Response: {response}")
        
        # Test multi-turn conversation
        response2 = bedrock.chat("What about 3 + 3?")
        print(f"✓ Follow-up response: {response2}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}")
        print(f"  → {e}")
    
    print("\n" + "=" * 60)
