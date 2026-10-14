"""
Claude API integration for PayPal Hackathon project.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ClaudeAPI:
    """Wrapper for Claude API interactions."""

    def __init__(self, model: str = "claude-opus-5"):
        """
        Initialize Claude API client.
        
        Args:
            model: Claude model to use (default: claude-opus-5)
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env file or as environment variable."
            )
        
        # Get workspace ID (optional, required for non-workspace-scoped keys)
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
        
        # Initialize with workspace ID if provided
        if workspace_id:
            self.client = Anthropic(
                api_key=api_key,
                default_headers={"anthropic-workspace-id": workspace_id}
            )
        else:
            self.client = Anthropic(api_key=api_key)
            
        self.model = model
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Claude's response text
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


if __name__ == "__main__":
    # Example usage
    claude = ClaudeAPI()
    
    print("Claude API initialized successfully!")
    print(f"Model: {claude.model}")
    print("\nTesting API call...")
    
    try:
        response = claude.chat("What is 2 + 2?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"Details: {e}")
        print("\n⚠️  Troubleshooting:")
        print("1. Verify your workspace ID is correct")
        print("2. Ensure the API key belongs to that workspace")
        print("3. Check that the workspace is active in console.anthropic.com")
