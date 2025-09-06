# LMCache Trace Replayer

A lightweight tool for replaying LLM request traces to benchmark server performance, designed for testing chat completion endpoints.

## Features

- **Chat Completions Focus**: Specialized support for `/v1/chat/completions` endpoint
- **Accurate Timing**: Maintains original request timing and arrival patterns  
- **Real-time Metrics**: Outputs detailed performance metrics to CSV
- **Simple Configuration**: YAML-based configuration
- **Streaming Support**: Measures Time-to-First-Token (TTFT) accurately

## Quick Start

### 1. Install Dependencies

```bash
pip install openai pydantic pyyaml pandas
```

### 2. Start vLLM Server

**Important**: Make sure to note the model name and port for configuration.

```bash
# Example: Start vLLM server with Llama-3 model
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --served-model-name llama-3-instruct \
    --host 127.0.0.1 \
    --port 8000
```

### 3. Configure the Tool

NOTE: Ensure model names and ports match exactly between:
- vLLM server (`--served-model-name`)
- Configuration file (`server.model`)  
- Trace file (`body.model` in requests)

Edit `config.yaml`:

```yaml
server:
  base_url: "http://localhost:8000/v1"  # Match vLLM server port
  api_key: "EMPTY"
  model: "llama-3-instruct"  # Match vLLM --served-model-name

replay_config:
  trace_file: "test_chat_trace.jsonl"
  max_duration: 60.0
  output_file: "summary.csv"
```

### 4. Run the Replayer

```bash
python trace_replayer.py --config config.yaml
```

## Usage Examples

### Basic Usage
```bash
# Use default config.yaml
python trace_replayer.py

# Specify custom config
python trace_replayer.py --config my_config.yaml

# Override trace file
python trace_replayer.py --trace_file custom_trace.jsonl

# Set maximum duration
python trace_replayer.py --max_duration 30

# Override output file
python trace_replayer.py --output custom_results.csv

# Combine multiple overrides
python trace_replayer.py --config my_config.yaml --trace_file custom.jsonl --output results.csv
```

### Analyzing Results with Python
```python
import pandas as pd

# Load results
df = pd.read_csv('summary.csv')

# Calculate basic metrics
success_rate = df['is_success'].mean()
avg_ttft = df[df['is_success']]['ttft'].mean()
total_requests = len(df)

print(f"Success Rate: {success_rate:.1%}")
print(f"Average TTFT: {avg_ttft:.3f}s")
print(f"Total Requests: {total_requests}")

# Performance analysis
print(f"Input Tokens: {df['input_token_len'].sum()}")
print(f"Output Tokens: {df['output_token_len'].sum()}")
print(f"Total Duration: {df['finish_time'].max() - df['launch_time'].min():.2f}s")
```

### Model Name Consistency Check
```bash
# 1. Check your vLLM server model name
curl http://localhost:8000/v1/models

# 2. Verify config.yaml matches
grep "model:" config.yaml

# 3. Check trace file model names  
grep -o '"model":"[^"]*"' test_chat_trace.jsonl | head -5

# All three should show the same model name!
```

## Trace File Format

The trace file must be in JSONL (JSON Lines) format. **Current MVP version only supports Chat Completions requests.**

### Format Requirements

1. **First line**: Metadata object with trace information
2. **Subsequent lines**: Chat completion request objects, one per line
3. **File extension**: `.jsonl`
4. **Encoding**: UTF-8

### Required Fields

**Metadata line (first line)**:
```json
{"trace_version": "0.1", "created_at": "2025-09-06T12:00:00Z", "description": "Optional description"}
```

**Chat completion request lines**:
```json
{"timestamp": 1725624000.0, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3-instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}, "request_id": "req-001"}
```

### Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | float | Yes | Unix timestamp in seconds (with decimals) |
| `method` | string | Yes | HTTP method, typically "POST" |
| `url` | string | Yes | Must be "/v1/chat/completions" |
| `body` | object | Yes | Complete chat completion request payload |
| `body.model` | string | Yes | **Must match vLLM server model name** |
| `request_id` | string | Yes | Unique identifier for the request |

### Example Complete Trace File

```jsonl
{"trace_version": "0.1", "created_at": "2025-09-06T12:00:00Z", "description": "Chat completions test trace"}
{"timestamp": 1725624000.0, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3-instruct", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 50}, "request_id": "chat-req-001"}
{"timestamp": 1725624000.8, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-3-instruct", "messages": [{"role": "user", "content": "What is AI?"}], "max_tokens": 100}, "request_id": "chat-req-002"}
```

### Supported Endpoints

- ✅ `/v1/chat/completions` - Chat-style conversations (MVP focus)
- ❌ `/v1/completions` - Will be supported in future versions
- ❌ `/v1/embeddings` - Will be supported in future versions

**Note**: Non-chat completion requests in trace files will be automatically filtered out and skipped.

### Timing and Order

- Requests are replayed based on relative timestamps
- First request starts at time 0, others follow relative delays
- Original timestamp values are preserved for reference

## Output Metrics

The CSV output contains per-request metrics:

| Column | Description |
|--------|-------------|
| `request_id` | Request identifier |
| `ttft` | Time to First Token (seconds) |
| `input_token_len` | Input tokens count |
| `output_token_len` | Output tokens count |
| `launch_time` | Request start timestamp |
| `finish_time` | Request end timestamp |
| `is_success` | Request completion status |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check if vLLM server is running on specified port |
| Model not found (404) | **Ensure model names match**: vLLM server, config.yaml, and trace file |
| Port connection failed | **Verify port consistency** between vLLM server and config.yaml |
| File not found | Verify `trace_file` path in config |
| YAML parse error | Validate `config.yaml` syntax |
| Memory issues | Reduce `max_duration` or `max_concurrent` |
| Requests filtered out | Check trace file contains `/v1/chat/completions` requests only |

### Common Model Name Issues

```bash
# Problem: vLLM server shows "model not found" error
# Solution: Check model name consistency

# 1. Check what model vLLM server is serving:
curl http://localhost:8000/v1/models

# 2. Expected response should match your config:
# {"data": [{"id": "llama-3-instruct", ...}]}

# 3. Update config.yaml if names don't match:
server:
  model: "llama-3-instruct"  # Use the exact name from step 1
```

## Command Line Options

```
usage: trace_replayer.py [-h] [--config CONFIG] [--trace_file TRACE_FILE] 
                        [--max_duration MAX_DURATION] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config file (default: config.yaml)
  --trace_file TRACE_FILE
                        Override trace file from config
  --max_duration MAX_DURATION
                        Override max duration from config
  --output OUTPUT       Override output file from config
```

## Example Workflow

### Complete Setup Example

```bash
# 1. Start vLLM server (in terminal 1)
conda activate vllm
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/llama-3-instruct \
    --served-model-name llama-3-instruct \
    --host 127.0.0.1 \
    --port 8000

# 2. Verify server is running (in terminal 2)
curl http://localhost:8000/v1/models
# Should return: {"data": [{"id": "llama-3-instruct", ...}]}

# 3. Configure trace replayer
cat > config.yaml << EOF
server:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  model: "llama-3-instruct"  # match step 1

replay_config:
  trace_file: "test_chat_trace.jsonl"
  max_duration: 60.0
  output_file: "summary.csv"
EOF

# 4. Run trace replayer
python trace_replayer.py --config config.yaml

# 5. Check results
cat summary.csv
```

### Expected Output

```
============================================================
TRACE REPLAY SUMMARY
============================================================
Total Requests: 3
Successful Requests: 3 (100.00%)
Total Duration: 2.56s
Average TTFT: 0.2770s
Total Input Tokens: 130
Total Output Tokens: 143
Throughput: 1.17 req/s
Token Throughput: 55.91 tokens/s
============================================================
```

## Future Development

This is an MVP (Minimum Viable Product) focused on Chat Completions. Future versions will include:

- ✅ **Current**: `/v1/chat/completions` support
- 🔄 **Planned**: `/v1/completions` endpoint support  
- 🔄 **Planned**: `/v1/embeddings` endpoint support
- 🔄 **Planned**: LMCache integration and caching performance testing
- 🔄 **Planned**: Advanced filtering and configuration options
- 🔄 **Planned**: Trace Recorder for realistic trace data
- 🔄 **Planned**: Real-time performance monitoring dashboard
