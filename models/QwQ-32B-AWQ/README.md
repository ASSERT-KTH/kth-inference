# QwQ-32B-AWQ

```bash
bash models/QwQ-32B-AWQ/serve.sh [model_name]   # default Qwen/QwQ-32B-AWQ, vLLM 0.15.1
```

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
