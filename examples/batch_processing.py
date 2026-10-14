"""
Batch processing example using Claude API.
"""

import sys
sys.path.insert(0, '..')

from claude_api import ClaudeAPI


def process_items(items: list) -> list:
    """
    Process a batch of items with Claude.
    
    Args:
        items: List of strings to process
        
    Returns:
        List of responses from Claude
    """
    claude = ClaudeAPI()
    results = []
    
    for i, item in enumerate(items, 1):
        print(f"Processing {i}/{len(items)}: {item[:50]}...")
        response = claude.chat(f"Summarize this in one sentence: {item}")
        results.append({
            "input": item,
            "output": response
        })
    
    return results


def main():
    """Run batch processing example."""
    items = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a high-level programming language",
        "The Earth revolves around the Sun"
    ]
    
    results = process_items(items)
    
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    
    for result in results:
        print(f"\nInput: {result['input']}")
        print(f"Output: {result['output']}")


if __name__ == "__main__":
    main()
