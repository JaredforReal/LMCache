# LMCache Trace Replayer

A lightweight tool for replaying LLM request traces to benchmark server performance, particularly for testing caching systems like LMCache.

## Features

- **Multi-endpoint Support**: `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`
- **Accurate Timing**: Maintains original request timing and arrival patterns
- **Real-time Metrics**: Outputs detailed performance metrics to CSV
- **Simple Configuration**: YAML-based configuration

## Quick Start

### 1. Install Dependencies

```bash
pip install openai pydantic pyyaml pandas
```

### 2. Configure the Tool

Edit `config.yaml`:

```yaml
server:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  model: "meta-llama/Llama-3.1-8B-Instruct"

replay_config:
  trace_file: "test_basic_trace.jsonl"
  max_duration: 60.0
```

### 3. Start LLM Server

**Baseline (vLLM only):**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**With LMCache:**
```bash
LMCACHE_CONFIG_FILE=lmcache.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct 
    --port 8000 
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
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

# Analyze by endpoint
endpoint_stats = df.groupby('endpoint').agg({
    'ttft': 'mean',
    'is_success': 'mean',
    'input_token_len': 'mean',
    'output_token_len': 'mean'
}).round(3)
print(endpoint_stats)
```

### Benchmarking Workflow
```bash
# 1. Run baseline (no caching)
python trace_replayer.py --config baseline_config.yaml
mv summary.csv baseline_results.csv

# 2. Run with LMCache
python trace_replayer.py --config lmcache_config.yaml  
mv summary.csv lmcache_results.csv

# 3. Compare results
python -c "
import pandas as pd
baseline = pd.read_csv('baseline_results.csv')
lmcache = pd.read_csv('lmcache_results.csv')

print('Baseline TTFT:', baseline[baseline['is_success']]['ttft'].mean())
print('LMCache TTFT:', lmcache[lmcache['is_success']]['ttft'].mean())
print('Improvement:', 
      (baseline[baseline['is_success']]['ttft'].mean() - 
       lmcache[lmcache['is_success']]['ttft'].mean()) / 
       baseline[baseline['is_success']]['ttft'].mean() * 100, '%')
"
```

## Trace File Format

The trace file must be in JSONL (JSON Lines) format following the `VllmTrace` specification:

### Format Requirements

1. **First line**: Metadata object with trace information
2. **Subsequent lines**: Individual request objects, one per line
3. **File extension**: `.jsonl`
4. **Encoding**: UTF-8

### Required Fields

**Metadata line (first line)**:
```json
{"trace_version": "0.1", "created_at": "2025-09-06T12:00:00Z", "description": "Optional description"}
```

**Request lines**:
```json
{"timestamp": 1725624000.0, "method": "POST", "url": "/v1/chat/completions", "body": {...}, "request_id": "req-001"}
```

### Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | float | Yes | Unix timestamp in seconds (with decimals) |
| `method` | string | Yes | HTTP method, typically "POST" |
| `url` | string | Yes | API endpoint path |
| `body` | object | Yes | Complete request payload |
| `request_id` | string | Yes | Unique identifier for the request |

### Example Complete Trace File

```jsonl
{"trace_version": "0.1", "created_at": "2025-09-06T12:00:00Z", "description": "Test trace"}
{"timestamp": 1725624000.0, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]}, "request_id": "req-001"}
{"timestamp": 1725624000.5, "method": "POST", "url": "/v1/completions", "body": {"model": "my-model", "prompt": "Hello"}, "request_id": "req-002"}
{"timestamp": 1725624001.0, "method": "POST", "url": "/v1/embeddings", "body": {"model": "my-model", "input": "Hello"}, "request_id": "req-003"}
```

### Supported Endpoints

- `/v1/chat/completions` - Chat-style conversations
- `/v1/completions` - Text completion
- `/v1/embeddings` - Text embeddings

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

## Configuration Options

```yaml
# Server connection
server:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"        # Use "EMPTY" for local vLLM
  model: "meta-llama/Llama-3.1-8B-Instruct"

# Replay settings
replay_config:
  trace_file: "test_basic_trace.jsonl"
  max_duration: 60.0      # Stop after N seconds
  output_file: "summary.csv"  # Output CSV filename

# Advanced settings (optional)
advanced:
  request_timeout: 30.0   # Request timeout in seconds
  max_concurrent: 10      # Max concurrent requests
  enable_retry: false     # Enable retry on failures
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check if LLM server is running on specified port |
| File not found | Verify `trace_file` path in config |
| YAML parse error | Validate `config.yaml` syntax |
| Memory issues | Reduce `max_duration` or `concurrent_limit` |
| SSL errors | Use `http://` instead of `https://` for local servers |

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

## Real-world Testing

### Start vLLM Server
```bash
# In another terminal, start vLLM server
python -m vllm.entrypoints.openai.api_server --model facebook/opt-350m --port 8001
```

### Model Compatibility
- `facebook/opt-350m`: Supports completions API only
- For embeddings: Use models with embedding support
- For chat completions: Use chat-tuned models

### Example Test
```bash
python trace_replayer.py --config test_config_simple.yaml --trace_file test_clean_trace.jsonl --output results.csv
```

Expected output: 100% success rate with detailed metrics in CSV format.
