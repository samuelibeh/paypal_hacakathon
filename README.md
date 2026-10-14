# PayPal Hackathon - Claude API Integration

This project integrates the Claude API for AI-powered functionality.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
# Optional: if your key requires workspace ID
ANTHROPIC_WORKSPACE_ID=your_workspace_id_here
```

**Getting your Workspace ID:**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Navigate to your workspace settings
3. Copy the Workspace ID
4. Add it to your `.env` file as `ANTHROPIC_WORKSPACE_ID`

### 3. Usage

**Basic chat example:**
```python
from claude_api import ClaudeAPI

claude = ClaudeAPI()
response = claude.chat("What is 2 + 2?")
print(response)
```

**Run the example:**
```bash
python3 claude_api.py
```

## API Reference

### ClaudeAPI

Main class for interacting with Claude API.

**Methods:**
- `chat(user_message: str) -> str` - Send a message and get a response
- `reset_conversation()` - Clear conversation history

**Example:**
```python
from claude_api import ClaudeAPI

claude = ClaudeAPI(model="claude-opus-5")
response = claude.chat("Hello!")
print(response)

# Multi-turn conversation
response2 = claude.chat("What's your name?")
print(response2)

# Reset conversation
claude.reset_conversation()
```

## Available Models

- `claude-opus-5` (default) - Most capable
- `claude-sonnet-5-5` - Balanced performance/speed
- `claude-haiku-4-5-20251001` - Fastest, smallest

## Security

⚠️ **Never commit `.env` file to git.** It's in `.gitignore` by default.

When you rotate your API key:
1. Delete the old key from [console.anthropic.com](https://console.anthropic.com/account/keys)
2. Generate a new key
3. Update your `.env` file

## License

MIT
