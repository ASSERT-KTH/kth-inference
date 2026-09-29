#!/usr/bin/env python3
import requests
import argparse
import sys
import re
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_model(api_base):
    try:
        response = requests.get(f"{api_base}/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")
    return None

def test_token_limit(api_url, model, n_tokens):
    """Returns (Success, ErrorMessage)"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": n_tokens,
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            return True, None
        else:
            try:
                error_msg = response.json().get("error", {}).get("message", "")
            except:
                error_msg = response.text
            return False, error_msg
    except Exception as e:
        return False, str(e)

def parse_limit_from_error(error_msg):
    # vLLM often returns: "This model's maximum context length is 32768 tokens. However, you requested..."
    match = re.search(r"maximum context length is (\d+)", error_msg)
    if match:
        return int(match.group(1))
    return None

def find_limit_binary_search(api_url, model, min_val, max_val):
    console.print(f"[yellow]Starting binary search between {min_val} and {max_val}...[/yellow]")
    last_success = min_val
    
    while min_val <= max_val:
        mid = (min_val + max_val) // 2
        console.print(f"  Testing {mid} tokens...", end="\r")
        success, _ = test_token_limit(api_url, model, mid)
        
        if success:
            last_success = mid
            min_val = mid + 1
        else:
            max_val = mid - 1
    
    console.print()  # New line after progress
    return last_success

def main():
    parser = argparse.ArgumentParser(description="Check the empirical max_tokens limit of a vLLM server.")
    parser.add_argument("--api", type=str, default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--model", type=str, help="Model name (optional, will fetch if not provided)")
    args = parser.parse_args()

    api_base = args.api.rstrip("/")
    api_url = f"{api_base}/chat/completions"

    model = args.model or get_model(api_base)
    if not model:
        console.print("[bold red]Could not determine model name. Is the server running?[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[bold green]Probing Model:[/bold green] {model}\n[bold green]Endpoint:[/bold green] {api_url}"))

    # Initial probe with a very high value to see if vLLM tells us the limit
    console.print("[yellow]Probing for limit via error message...[/yellow]")
    success, error_msg = test_token_limit(api_url, model, 1_000_000)
    
    if success:
        console.print("[bold green]Success at 1,000,000 tokens! The limit might be higher than expected.[/bold green]")
        limit = 1_000_000
    else:
        limit = parse_limit_from_error(error_msg)
        if limit:
            console.print(f"[bold cyan]Detected limit from server error message: {limit}[/bold cyan]")
        else:
            console.print("[yellow]Could not extract limit from error. Falling back to binary search.[/yellow]")
            # Binary search from 1 to 256k
            limit = find_limit_binary_search(api_url, model, 1, 262144)

    console.print(Panel(f"[bold white]Empirical max_tokens limit for [bold cyan]{model}[/bold cyan]:[/bold white]\n[bold green]{limit}[/bold green]", border_style="green"))

if __name__ == "__main__":
    main()
