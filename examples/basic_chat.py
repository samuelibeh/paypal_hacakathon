"""
Basic chat example using Claude API.
"""

import sys
sys.path.insert(0, '..')

from claude_api import ClaudeAPI


def main():
    """Run basic chat example."""
    claude = ClaudeAPI()
    
    print("Claude API Chat Example")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = claude.chat(user_input)
        print(f"Claude: {response}\n")


if __name__ == "__main__":
    main()
