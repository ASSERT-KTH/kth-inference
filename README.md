# Simple VLLM serving for fast inference on a single H100

Perfect for use on KTH's DGX H100 since it mostly sits idle ¯\\_(ツ)_/¯

## Quick Start

1. Install the dependencies
```bash
pip install .
```

2. Start the vLLM server:
```bash
sh serve.sh
```

3. In a separate terminal, start the chat interface:
```bash
uv run chat.py
```

## Configuration

### Chat Interface

You can customize the chat interface with these parameters:

```bash
# Set a custom system prompt
uv run chat.py --system "You are a Python programming expert"
```

### Model Configuration

You can specify a different model when starting the server:

```bash
# Use a different model
sh serve.sh Qwen/Qwen2.5-32B-Instruct-AWQ
```

The default model is `Qwen/QwQ-32B-AWQ`.

> **Warning**: The server script is configured for AWQ models with Marlin quantization. If you want to use a non-AWQ model, you'll need to modify the `serve.sh` script to remove or change the `--quantization awq_marlin` parameter.

## Usage

- Type your message and press Enter for new lines
- Press Ctrl+J or ESC+Enter to send the message
- Supports pasting multi-line text

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
  
