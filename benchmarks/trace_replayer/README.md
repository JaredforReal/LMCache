# LMCache Trace Replayer & Generator Benchmark

## Overview

This benchmark is a unified tool designed to simulate realistic LLM workloads for performance evaluation, especially for caching systems like LMCache. It integrates both trace data generation and trace replaying into a single, configuration-driven script (`benchmark.py`).

Instead of using simplistic, repetitive prompts, this tool generates and replays diverse, conversational data to provide a more accurate assessment of an LLM server's performance under real-world conditions.

## Features

- **Unified Workflow**: Generate trace data and replay it using a single script and configuration file.
- **Configuration-Driven**: Easily switch between modes and parameters by editing `config.yaml`.
- **Realistic Data Generation**: Supports multiple generation modes:
  - `synthetic`: General-purpose conversational data.
  - `cache-focused`: Data specifically designed to test prefix caching effectiveness (e.g., for RAG).
  - `sharegpt`: Convert real-world ShareGPT datasets into a replayable trace.
- **Accurate Replay**: Replays requests while maintaining the original timing and arrival patterns of the trace.
- **Real-time Results**: Outputs detailed performance metrics to a CSV file during execution for live monitoring.
- **Analysis Script**: Includes a separate script (`analyze_results.py`) to compare and visualize results from different benchmark runs.

## Quick Start

The entire benchmark workflow is controlled by `benchmark.py` and `config.yaml`.

### 1. Configure `config.yaml`

First, open `config.yaml` and set it up for your desired task.

**Example: To generate a cache-focused trace:**

```yaml
# In config.yaml
mode: generate

generate_config:
  output_file: "cache_trace.jsonl"
  generation_mode: "cache-focused"
  num_requests: 200
  duration: 120.0
  cache_hit_ratio: 0.7
  seed: 42
```

**Example: To replay the generated trace:**

```yaml
# In config.yaml
mode: replay

server:
  base_url: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.1-8B-Instruct"

replay_config:
  trace_file: "cache_trace.jsonl"
  max_duration: 120.0
```

_(See the "Configuration (`config.yaml`)" section below for a full explanation of all options.)_

### 2. Start the LLM Server

Choose one of the following options.

**Baseline (vLLM without caching):**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**With LMCache:**

```bash
cd LMCache/benchmarks/trace_replayer/

LMCACHE_CONFIG_FILE=lmcache.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### 3. Run the Benchmark

Execute the main script. It will automatically pick up the settings from `config.yaml`.

```bash
# This one command runs either generation or replay based on the 'mode' in config.yaml
python benchmark.py
```

The script will:

1. Read `config.yaml`.
2. If `mode` is `generate`, it creates the trace file.
3. If `mode` is `replay`, it connects to the LLM server, replays the specified trace, and saves results to a new `summary-*.csv` file.

### 4. Analyze the Results

Use the `analyze_results.py` script to compare different runs (e.g., baseline vs. LMCache).

```bash
# Compare two specific result files
python analyze_results.py --pattern "baseline.csv,lmcache.csv" --plot

# Or use a glob pattern to compare all recent runs
python analyze_results.py --pattern "summary-*.csv" --plot
```

This will print a detailed performance comparison to the console and save visualization plots (like TTFT distribution) to the `plots/` directory.

## Configuration (`config.yaml`)

This file is the control center for the benchmark.

```yaml
# LMCache Benchmark Tool Configuration

# Execution mode: 'generate' or 'replay'
mode: replay

# Directory Configuration
directories:
  trace_dir: "traces"
  summary_dir: "summaries"

# LLM Server Configuration
server:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  model: "meta-llama/Llama-3.1-8B-Instruct"

# Trace Replay Configuration (used when mode is 'replay')
replay_config:
  # Filename of the trace to be replayed (will be looked for in `trace_dir`)
  trace_file: "trace.jsonl"
  # Maximum duration of the replay in seconds
  max_duration: 60.0

# Trace Generation Configuration (used when mode is 'generate')
generate_config:
  # Output filename for the generated trace (will be saved in `trace_dir`)
  output_file: "trace.jsonl"

  # Generation mode: 'synthetic', 'sharegpt', or 'cache-focused'
  generation_mode: "synthetic"
...
- `is_success`: Whether the request completed successfully.

## Troubleshooting

- **Connection Errors**: Ensure the LLM server is running and the `base_url` in `config.yaml` is correct.
- **`FileNotFoundError`**: Make sure the `trace_file` specified in `replay_config` exists inside the directory specified by `trace_dir`. You may need to run in `generate` mode first.
- **YAML Errors**: Check `config.yaml` for syntax errors.
- **Memory Issues**: If you encounter out-of-memory errors, try reducing `num_requests` or `duration` in your generation config.


  # Random seed for reproducibility
  seed: 42

  # --- Parameters for 'synthetic', 'sharegpt', and 'cache-focused' modes ---
  num_requests: 100
  duration: 60.0

  # --- Parameters for 'sharegpt' mode ---
  # Path to the input ShareGPT-formatted JSON file
  input_file: "sharegpt_data.json" # Required if generation_mode is 'sharegpt'

  # --- Parameters for 'cache-focused' mode ---
  # Target cache hit ratio (e.g., 0.7 for 70% hits)
  cache_hit_ratio: 0.7
```

## Benchmarking LMCache Effectiveness

To properly evaluate LMCache, follow these steps:

### Step 1: Generate a Cache-Focused Trace

Configure `config.yaml` to generate a trace designed to test caching.

```yaml
# config.yaml
mode: generate
generate_config:
  output_file: "cache_test_trace.jsonl"
  generation_mode: "cache-focused"
  num_requests: 200
  duration: 120.0
  cache_hit_ratio: 0.8
```

Run the script to create the trace file:

```bash
python benchmark.py
```

### Step 2: Run Baseline (No Caching)

1.  Start the vLLM server **without** LMCache.
2.  Configure `config.yaml` to replay the trace.
    ```yaml
    # config.yaml
    mode: replay
    replay_config:
      trace_file: "cache_test_trace.jsonl"
    ```
3.  Run the benchmark.
    ```bash
    python benchmark.py
    ```
4.  Rename the output file for clarity.
    ```bash
    mv summary-*.csv baseline_results.csv
    ```

### Step 3: Run with LMCache

1.  Stop the baseline server and restart it **with** LMCache enabled.
2.  Using the same `config.yaml` as Step 2, run the benchmark again.
    ```bash
    python benchmark.py
    ```
3.  Rename the new output file.
    ```bash
    mv summary-*.csv lmcache_results.csv
    ```

### Step 4: Compare Results

Use the analysis script to see the performance difference.

```bash
python analyze_results.py --pattern "baseline_results.csv,lmcache_results.csv" --plot
```

Look for:

- **Lower TTFT (Time to First Token)** with LMCache.
- **Higher Throughput (req/s and tokens/s)** with LMCache.

## Output and Analysis

The benchmark outputs:

1.  **Trace File (`.jsonl`)**: If `mode` is `generate`.
2.  **Results CSV (`summary-*.csv`)**: If `mode` is `replay`. Contains per-request metrics.
3.  **Console Summary**: A summary is printed to the console after a replay finishes.
4.  **Analysis Plots**: The `analyze_results.py` script generates plots in the `plots/` directory.

**CSV Columns:**

- `chat_id`: Unique request identifier.
- `ttft`: Time to First Token (in seconds). This is a key metric for user experience.
- `input_token_len` / `output_token_len`: Number of tokens in the prompt and response.
- `launch_time` / `finish_time`: Monotonic timestamps for calculating throughput.
- `is_success`: Whether the request completed successfully.

## Troubleshooting

- **Connection Errors**: Ensure the LLM server is running and the `base_url` in `config.yaml` is correct.
- **`FileNotFoundError`**: Make sure the `trace_file` specified in `replay_config` exists. You may need to run in `generate` mode first.
- **YAML Errors**: Check `config.yaml` for syntax errors.
- **Memory Issues**: If you encounter out-of-memory errors, try reducing `num_requests` or `duration` in your generation config.
