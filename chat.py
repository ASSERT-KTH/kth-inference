#!/usr/bin/env python3
"""
Enhanced CLI chat interface for LLMs using vLLM's OpenAI-compatible API.
Features a scrollable conversation history and prominent tokens/s display.
"""

import os
import json
import time
import requests
import argparse
import shutil
from datetime import datetime

from rich import box
from rich.box import ROUNDED
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.prompt import PromptBase, Prompt
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings


# Create console
console = Console()

def print_header(model_name):
    """Print a nice header for the chat interface."""
    # Extract just the model name without the organization
    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    console.print()
    console.print(Panel(
        f"[bold cyan]{display_name} Chat Interface[/bold cyan]",
        subtitle="[dim]Type 'help' for available commands[/dim]",
        box=ROUNDED,
        expand=False,
        border_style="cyan"
    ))
    console.print(
        "[dim]- Press [bold]Enter[/bold] to send your message\n"
        "- Press [bold]Alt+Enter[/bold] to add a new line\n"
        "- Type 'exit' to quit, 'clear' to reset conversation\n"
        "- Type 'help' for more commands[/dim]"
    )
    console.print()

def get_available_models(api_base_url):
    """
    Get the list of available models from the API.
    
    Args:
        api_base_url: Base URL of the OpenAI-compatible API
    
    Returns:
        List of available model IDs, or None if the request failed
    """
    try:
        # Extract the base URL (without the /v1/chat/completions part)
        base_url = api_base_url.split('/v1/')[0]
        models_url = f"{base_url}/v1/models"
        
        response = requests.get(models_url)
        if response.status_code == 200:
            models_data = response.json()
            if 'data' in models_data:
                return [model['id'] for model in models_data['data']]
        return None
    except Exception as e:
        console.print(f"[bold red]Error fetching available models: {str(e)}[/bold red]")
        return None

def wait_for_server(api_base_url, max_retries=10, retry_delay=2):
    """
    Wait for the server to become available.
    
    Args:
        api_base_url: Base URL of the OpenAI-compatible API
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
    
    Returns:
        True if the server is available, False otherwise
    """
    console.print("[yellow]Checking if vLLM server is running...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[yellow]Connecting to server..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Connecting", total=max_retries)
        
        for i in range(max_retries):
            try:
                # Extract the base URL (without the /v1/chat/completions part)
                base_url = api_base_url.split('/v1/')[0]
                models_url = f"{base_url}/v1/models"
                
                response = requests.get(models_url)
                if response.status_code == 200:
                    console.print("[bold green]Server is running![/bold green]")
                    return True
                
                progress.update(task, advance=1, description=f"[yellow]Waiting for server (attempt {i+1}/{max_retries})...")
                time.sleep(retry_delay)
            except requests.exceptions.ConnectionError:
                progress.update(task, advance=1, description=f"[yellow]Server not ready (attempt {i+1}/{max_retries})...")
                time.sleep(retry_delay)
    
    console.print("[bold red]Server did not become available after {max_retries} attempts.[/bold red]")
    console.print("[yellow]Make sure the vLLM server is running with: ./start.sh[/yellow]")
    return False

class Message:
    """Representation of a chat message."""
    
    def __init__(self, role, content="", timestamp=None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        """Convert to dict for API calls."""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def to_panel(self, width=None):
        """Convert message to a Rich Panel for display."""
        if self.role == "system":
            return Panel(
                self.content,
                title="[blue]System[/blue]",
                title_align="left",
                border_style="blue",
                padding=(0, 1),  # Reduced padding
                width=width,
                box=box.SIMPLE  # Simpler box style
            )
        elif self.role == "user":
            return Panel(
                self.content,
                title="[blue]You[/blue]",
                title_align="left",
                border_style="blue",
                padding=(0, 1),  # Reduced padding
                width=width,
                box=box.SIMPLE  # Simpler box style
            )
        elif self.role == "assistant":
            return Panel(
                self.content,
                title="[green]Assistant[/green]",
                title_align="left",
                border_style="green",
                padding=(0, 1),  # Reduced padding
                width=width,
                box=box.SIMPLE  # Simpler box style
            )
        else:
            return Panel(
                self.content,
                title=f"[yellow]{self.role.capitalize()}[/yellow]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),  # Reduced padding
                width=width,
                box=box.SIMPLE  # Simpler box style
            )

class StatsBar:
    """Stats bar for displaying generation metrics prominently."""
    
    def __init__(self):
        self.token_count = 0
        self.tokens_per_second = 0
        self.total_time = 0.0
        self.is_generating = False
        self.status = "Ready"
    
    def update(self, token_count=None, tokens_per_second=None, total_time=None, 
               is_generating=None, status=None):
        """Update stats with new values."""
        if token_count is not None:
            self.token_count = token_count
        if tokens_per_second is not None:
            self.tokens_per_second = tokens_per_second
        if total_time is not None:
            self.total_time = total_time
        if is_generating is not None:
            self.is_generating = is_generating
        if status is not None:
            self.status = status
    
    def _get_speed_style(self):
        """Get the appropriate style based on tokens/s speed."""
        if self.tokens_per_second >= 70:
            return "bold bright_green", "🚀"  # Rocket for very fast
        elif self.tokens_per_second >= 40:
            return "bold green", "⚡"  # Lightning for fast
        elif self.tokens_per_second >= 20:
            return "bold yellow", "🔆"  # Sun for medium
        else:
            return "bold orange3", "🔸"  # Diamond for slower
    
    def to_renderable(self):
        """Convert to a Rich renderable for display."""
        if self.is_generating:
            # Get style based on speed
            speed_style, speed_icon = self._get_speed_style()
            
            # Create a more visually appealing tokens/s display
            tokens_per_second_text = Text(f" {self.tokens_per_second}", style=speed_style)
            
            # Add a larger font size effect using Unicode superscript/subscript
            tokens_per_second_text.append(" tokens/s", style="dim")
            
            # Create the main stats display
            stats = [
                Text(speed_icon, style=speed_style),
                tokens_per_second_text,
                Text(" | ", style="dim"),
                Text(f"{self.token_count} tokens", style="cyan"),
                Text(" | ", style="dim"),
                Text(f"{self.total_time:.1f}s", style="dim")
            ]
            
            # Create a progress bar representation of the speed
            max_speed = 100  # Maximum expected tokens/s
            speed_percentage = min(100, int((self.tokens_per_second / max_speed) * 100))
            
            # Create a visual bar
            bar_width = 20
            filled = int((speed_percentage / 100) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Create the complete display
            return Panel(
                Group(
                    Group(*stats),
                    Text(f"{bar} {speed_percentage}%", style=speed_style)
                ),
                border_style=speed_style.split()[1] if "bold" in speed_style else speed_style,
                padding=(0, 1),
                box=box.SIMPLE,
                title=f"[{speed_style}]Generating at {self.tokens_per_second} tokens/s[/{speed_style}]",
                title_align="left"
            )
        else:
            # Simple status display when idle
            return Panel(
                Text(self.status, style="cyan"),
                border_style="cyan",
                padding=(0, 1),
                box=box.SIMPLE
            )

class MultilinePrompt:
    """A prompt that supports multi-line input with Enter to submit and Alt+Enter for newlines."""
    
    def __init__(self):
        self.session = PromptSession()
        self.kb = KeyBindings()
        
        # Enter submits
        @self.kb.add('enter')
        def _(event):
            event.current_buffer.validate_and_handle()
        
        # Alt+Enter inserts a newline
        @self.kb.add('escape', 'enter')
        def _(event):
            event.current_buffer.insert_text('\n')
        
        # Keep Ctrl+J for compatibility
        @self.kb.add('c-j')
        def _(event):
            event.current_buffer.validate_and_handle()
    
    def get_input(self) -> str:
        """Get input from the user."""
        # Leave a clean space for input
        print()
        
        text = self.session.prompt(
            ">>> ",  # Simple prompt
            multiline=True,
            key_bindings=self.kb,
            prompt_continuation=lambda width, line_number, is_soft_wrap: '... '
        )
        return text.rstrip()

class ChatInterface:
    """Enhanced chat interface with scrollable conversation history and prominent tokens/s display."""
    
    def __init__(self, api_url, model_name, system_prompt=None):
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []
        self.visible_messages = []
        self.scroll_position = 0
        self.stats = StatsBar()
        self.start_time = 0
        self.total_tokens = 0
        self.total_time = 0
        self.current_response = ""
        self.prompt = MultilinePrompt()
        
        # Terminal size
        self.terminal_size = shutil.get_terminal_size()
        
        # Track if user has manually scrolled
        self.user_scrolled = False
        
        # Message height cache
        self.message_heights = {}
        
        # Set default max_tokens
        self.max_tokens = 4096
        
        # Add system message if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def _update_max_tokens(self):
        """Update max_tokens based on the current model name."""
        if self.model_name and "gpt-oss-20b" in self.model_name.lower():
            self.max_tokens = 32768
            console.print(f"[bold cyan]Detected gpt-oss-20b model. Setting max_tokens to {self.max_tokens} (32k context).[/bold cyan]")
        else:
            self.max_tokens = 4096
    
    def add_message(self, role, content):
        """Add a message to the conversation."""
        message = Message(role, content)
        self.messages.append(message)
        # Auto-scroll to bottom only if user hasn't manually scrolled
        if not self.user_scrolled:
            self.scroll_to_bottom()
        return message
    
    def estimate_message_height(self, message, width):
        """Estimate the height of a message in terminal lines."""
        # Check if we have a cached height for this message
        message_key = f"{message.role}:{message.content}:{width}"
        if message_key in self.message_heights:
            return self.message_heights[message_key]
        
        # Estimate height based on content length and width
        # Account for word wrapping by estimating chars per line
        chars_per_line = max(40, width - 10)  # Adjust for padding and borders
        
        # Count newlines in the content
        newlines = message.content.count('\n')
        
        # Estimate wrapped lines (content length / chars per line)
        wrapped_lines = len(message.content) // chars_per_line
        
        # Total height: wrapped lines + explicit newlines + overhead for panel borders and title
        estimated_height = wrapped_lines + newlines + 3  # +3 for panel overhead
        
        # Cache the result
        self.message_heights[message_key] = estimated_height
        return estimated_height
    
    def calculate_visible_messages(self):
        """Calculate which messages should be visible based on scroll position."""
        terminal_height = self.terminal_size.lines
        # Space available for messages (reserve space for stats bar and input area)
        available_height = max(10, terminal_height - 8)
        
        # Get terminal width for message formatting
        width = max(60, self.terminal_size.columns - 4)
        
        # If user hasn't manually scrolled, show the most recent messages
        if not self.user_scrolled:
            visible = []
            current_height = 0
            
            for message in reversed(self.messages):
                msg_height = self.estimate_message_height(message, width)
                if current_height + msg_height <= available_height:
                    visible.insert(0, message)  # Insert at beginning to maintain order
                    current_height += msg_height
                else:
                    # If we can't fit all messages, prioritize showing the latest
                    if not visible:
                        visible.insert(0, message)  # Always show at least one message
                    break
            
            return visible
        else:
            # User has manually scrolled - respect scroll position
            visible = []
            current_height = 0
            
            # Start from the scroll position
            for i in range(self.scroll_position, len(self.messages)):
                msg_height = self.estimate_message_height(self.messages[i], width)
                if current_height + msg_height <= available_height:
                    visible.append(self.messages[i])
                    current_height += msg_height
                else:
                    if not visible:
                        visible.append(self.messages[i])  # Always show at least one message
                    break
            
            return visible
    
    def scroll_to_bottom(self):
        """Scroll to the latest message."""
        self.user_scrolled = False
        if self.messages:
            self.scroll_position = 0  # Reset position - we'll show latest messages automatically
    
    def scroll_up(self):
        """Scroll conversation up."""
        if self.scroll_position > 0:
            self.user_scrolled = True
            self.scroll_position -= 1
    
    def scroll_down(self):
        """Scroll conversation down."""
        if self.scroll_position < len(self.messages) - 1:
            self.user_scrolled = True
            self.scroll_position += 1
        else:
            # If we're at the bottom, reset the user_scrolled flag
            self.user_scrolled = False
    
    def get_visible_messages(self):
        """Get the currently visible messages based on scroll position."""
        return self.calculate_visible_messages()
    
    def create_layout(self):
        """Create the layout for the chat interface."""
        layout = Layout()
        
        # Create a simpler layout with just chat area and stats bar
        # We'll handle input separately
        layout.split_column(
            Layout(name="chat_area", ratio=85),
            Layout(name="stats_bar", size=3)
        )
        
        # Initialize the chat area with an empty panel
        layout["chat_area"].update(
            Panel(
                Text("Welcome to the chat! Type your message when prompted."),
                title="Conversation",
                border_style="cyan",
                box=ROUNDED,
                padding=(1, 1)
            )
        )
        
        # Initialize the stats bar
        layout["stats_bar"].update(self.stats.to_renderable())
        
        return layout
    
    def update_chat_area(self, layout):
        """Update the chat area with current messages."""
        visible_messages = self.get_visible_messages()
        
        if not visible_messages:
            # Show a welcome message when there are no messages
            layout["chat_area"].update(
                Panel(
                    Text("Welcome to the chat! Type your message below."),
                    title="Conversation",
                    border_style="cyan",
                    box=ROUNDED,
                    padding=(1, 1)
                )
            )
            return
        
        # Get the width of the terminal for formatting
        width = max(60, self.terminal_size.columns - 4)
        
        # Create panels for each message
        message_panels = []
        for message in visible_messages:
            message_panels.append(message.to_panel(width=width))
        
        # Add scroll indicators if needed
        scroll_info = ""
        if self.scroll_position > 0 or (self.user_scrolled and visible_messages and visible_messages[0] != self.messages[0]):
            scroll_info += "[bold yellow]↑ More messages above[/bold yellow]\n"
        
        # Check if there are more messages below that aren't visible
        last_visible = visible_messages[-1] if visible_messages else None
        last_message = self.messages[-1] if self.messages else None
        
        if last_visible and last_message and last_visible != last_message:
            if scroll_info:
                scroll_info += "\n"
            scroll_info += "[bold yellow]↓ More messages below[/bold yellow]"
        
        # Group all the message panels with appropriate spacing
        messages_group = Group(*message_panels)
        
        # Create the final renderable with scroll indicators
        final_renderable = Group(
            Text(scroll_info) if scroll_info else Text(""),
            messages_group
        )
        
        # Update the chat area
        layout["chat_area"].update(
            Panel(
                final_renderable,
                title=f"Conversation ({len(self.messages)} messages)",
                border_style="cyan",
                box=ROUNDED,
                padding=(0, 1)
            )
        )
        
        # Ensure we're always showing the latest content during generation
        if self.stats.is_generating and not self.user_scrolled:
            self.scroll_to_bottom()
    
    def update_stats_bar(self, layout):
        """Update the stats bar with current stats."""
        layout["stats_bar"].update(self.stats.to_renderable())
    
    def update_input_area(self, layout, text="Type your message... (Ctrl+J to send)"):
        """Legacy method kept for compatibility."""
        # This method is no longer used as we're handling input differently
        pass
    
    def handle_user_command(self, command):
        """Handle special user commands."""
        command = command.lower()
        
        if command in ["exit", "quit"]:
            return "exit"
        elif command in ["clear", "reset"]:
            # Clear conversation history except system prompt
            if self.system_prompt:
                self.messages = [m for m in self.messages if m.role == "system"]
            else:
                self.messages = []
            self.scroll_position = 0
            self.stats.update(status="Conversation cleared")
            return "continue"
        elif command in ["help", "?"]:
            help_message = (
                "Available commands:\n"
                "- exit or quit: Exit the chat\n"
                "- clear or reset: Clear conversation history\n"
                "- help or ?: Show this help message\n"
                "- scroll up/down: Navigate through message history\n\n"
                "Input controls:\n"
                "- Press Enter to send your message\n"
                "- Press Alt+Enter to add a new line"
            )
            self.add_message("system", help_message)
            return "continue"
        elif command == "scroll up":
            self.scroll_up()
            return "continue"
        elif command == "scroll down":
            self.scroll_down()
            return "continue"
        
        # Not a special command
        return None
    
    def generate_response(self, layout, live):
        """Generate a response from the LLM."""
        # Update status
        self.stats.update(
            token_count=0, 
            tokens_per_second=0, 
            total_time=0.0, 
            is_generating=True,
            status="Generating..."
        )
        self.update_stats_bar(layout)
        live.refresh()
        
        # Reset counters
        self.start_time = time.time()
        self.current_response = ""
        
        # Prepare API request
        headers = {"Content-Type": "application/json"}
        
        # Convert messages to API format
        api_messages = [m.to_dict() for m in self.messages]
        
        
        data = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": True,
            "max_tokens": self.max_tokens
        }
        
        try:
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                stream=True
            )
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}"
                self.add_message("system", f"Error: {error_msg}")
                self.stats.update(is_generating=False, status="Error")
                self.update_stats_bar(layout)
                self.update_chat_area(layout)
                live.refresh()
                return
            
            # Add an empty assistant message that will be updated
            assistant_message = self.add_message("assistant", "")
            
            # Process the streaming response
            update_counter = 0
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    try:
                        json_data = json.loads(line[6:])
                        if 'choices' in json_data and json_data['choices'] and 'delta' in json_data['choices'][0]:
                            delta = json_data['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                self.current_response += content
                                
                                # Update the assistant message
                                assistant_message.content = self.current_response
                                
                                # Update token count and stats
                                self.stats.token_count += 1
                                elapsed_time = max(0.1, time.time() - self.start_time)
                                self.stats.tokens_per_second = int(self.stats.token_count / elapsed_time)
                                self.stats.total_time = elapsed_time
                                
                                # Update the display more frequently
                                update_counter += 1
                                if update_counter % 1 == 0:  # Update every token for smoother scrolling
                                    # During generation, always show the latest content
                                    self.scroll_to_bottom()
                                    self.update_stats_bar(layout)
                                    self.update_chat_area(layout)
                                    live.refresh()
                    except json.JSONDecodeError:
                        pass
            
            # Final update
            assistant_message.content = self.current_response
            self.total_tokens += self.stats.token_count
            self.total_time += self.stats.total_time
            
            # Update stats to show completion
            self.stats.update(
                is_generating=False,
                status=f"Generated {self.stats.token_count} tokens in {self.stats.total_time:.1f}s ({self.stats.tokens_per_second} tokens/s)"
            )
            
            # Force a final update to ensure message is fully visible
            self.scroll_to_bottom()
            self.update_stats_bar(layout)
            self.update_chat_area(layout)
            live.refresh()
            
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to the API server. Make sure it's running."
            self.add_message("system", f"Error: {error_msg}")
            self.stats.update(is_generating=False, status="Connection Error")
            self.update_stats_bar(layout)
            self.update_chat_area(layout)
            live.refresh()
        except Exception as e:
            error_msg = str(e)
            self.add_message("system", f"Error: {error_msg}")
            self.stats.update(is_generating=False, status="Error")
            self.update_stats_bar(layout)
            self.update_chat_area(layout)
            live.refresh()
    
    def run(self):
        """Run the chat interface."""
        # Check if the server is running
        if not wait_for_server(self.api_url):
            return
        
        # Get available models
        available_models = get_available_models(self.api_url)
        
        if available_models:
            console.print("[cyan]Available models:[/cyan]")
            for model in available_models:
                console.print(f"  - {model}")
            
            # If no model was specified, use the first available one
            if not self.model_name:
                self.model_name = available_models[0]
                console.print(f"[cyan]No model specified. Using the first available model: [bold]{self.model_name}[/bold][/cyan]")
            
            # Check if the requested model is available
            elif self.model_name not in available_models:
                console.print(f"[yellow]Warning: Model '{self.model_name}' not found in available models.[/yellow]")
                console.print(f"[yellow]Will try to use it anyway, but it might not work.[/yellow]")
        elif self.model_name:
            console.print(f"[yellow]Warning: Could not fetch model list. Proceeding with '{self.model_name}'.[/yellow]")
        else:
            console.print("[bold red]Error: No model specified and could not fetch available models from server.[/bold red]")
            return
        
        # Update max_tokens once the model name is resolved
        self._update_max_tokens()
        
        print_header(self.model_name)
        console.print("[dim italic]Watch the tokens/s display for real-time generation speed![/dim italic]")
        console.print()
        
        # Create the layout
        layout = self.create_layout()
        
        # Main chat loop with alternate display/input approach
        with Live(layout, refresh_per_second=20, console=console, auto_refresh=False) as live:
            while True:
                # Update terminal size and refresh the display
                self.terminal_size = shutil.get_terminal_size()
                self.update_chat_area(layout)
                self.update_stats_bar(layout)
                live.refresh()
                
                # Temporarily stop the live display
                live.stop()
                
                # Get user input
                console.print("[bold blue]Your message (Press Enter to send, Alt+Enter for new line):[/bold blue]")
                
                try:
                    # Use our improved MultilinePrompt
                    user_input = self.prompt.get_input()
                except KeyboardInterrupt:
                    console.print("[yellow]Chat session ended.[/yellow]")
                    return
                except Exception as e:
                    # Fallback to basic input if prompt_toolkit fails
                    console.print(f"[yellow]Advanced input failed: {str(e)}[/yellow]")
                    console.print("[yellow]Falling back to basic input. Type your message and press Enter:[/yellow]")
                    try:
                        user_input = input(" ")
                    except KeyboardInterrupt:
                        console.print("[yellow]Chat session ended.[/yellow]")
                        return
                
                # Restart the live display
                live.start()
                
                if not user_input:
                    continue
                
                # Check for special commands
                command_result = self.handle_user_command(user_input)
                
                if command_result == "exit":
                    self.stats.update(status="Goodbye!")
                    self.update_stats_bar(layout)
                    live.refresh()
                    time.sleep(1)  # Show goodbye message briefly
                    break
                elif command_result == "continue":
                    continue
                
                # Regular user message
                self.add_message("user", user_input)
                self.update_chat_area(layout)
                self.update_stats_bar(layout)
                live.refresh()
                
                # Generate response
                self.generate_response(layout, live)

def main():
    """Main function."""
    # Get model from environment if available
    env_model = os.environ.get("VLLM_MODEL")
    
    parser = argparse.ArgumentParser(description="Enhanced CLI chat interface for LLMs")
    parser.add_argument("--api", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL of the OpenAI-compatible API (default: http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--model", type=str, default=env_model,
                        help="Model name to use (default: fetched from /v1/models if not specified)")
    parser.add_argument("--system", type=str, 
                        default="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.",
                        help="System prompt to set the behavior of the assistant")
    
    args = parser.parse_args()
    
    try:
        # Create and run the chat interface
        chat_interface = ChatInterface(args.api, args.model, args.system)
        chat_interface.run()
    except KeyboardInterrupt:
        console.print(f"\n\n[yellow]Chat session ended by user.[/yellow]\n")

if __name__ == "__main__":
    main() 