# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import csv
import json
import logging
import time
import yaml
from pathlib import Path
from typing import List, Union, Optional, Any, TextIO
import openai
from pydantic import BaseModel, Field

# 导入生产版 API Server 支持的核心请求类型
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


class TraceRequest(BaseModel):
    """
    Represents a single request captured in a trace, containing the minimum
    necessary information for replaying.
    """

    timestamp: float = Field(
        ...,
        description=
        "Timestamp of when the request was received, in seconds since epoch. Essential for replaying traffic with original timing."
    )

    method: str = Field(..., description="The HTTP method, e.g., 'POST'.")

    url: str = Field(
        ...,
        description=
        "The request URL path, e.g., '/v1/chat/completions'. Crucial for directing the replayed request to the correct endpoint."
    )

    body: Union[ChatCompletionRequest, dict, Any] = Field(
        ...,
        description=
        "The request payload, parsed into the corresponding Pydantic model. This is the core content needed for the model to process the request."
    )

    request_id: str = Field(
        ...,
        description="A unique ID for this request in the trace for easier tracking."
    )


class TraceMetadata(BaseModel):
    """
    Metadata about the trace file.
    """
    # Version of the trace format.
    trace_version: str = "0.1"

    # Timestamp of when the trace was created, in ISO 8601 format.
    created_at: str = ""

    # Optional description of the trace's source or purpose.
    description: Optional[str] = None

    # The vLLM version used to generate the trace, for compatibility checks.
    vllm_version: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RequestResult(BaseModel):
    """
    Result of a single request execution.
    """
    request_id: str
    ttft: float  # Time to first token
    input_token_len: int
    output_token_len: int
    launch_time: float
    finish_time: float
    is_success: bool


# Constants
OUTPUT_FILENAME = "summary.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TraceReplayer:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_duration: float = 100.0,
        request_timeout: float = 30.0,
        max_concurrent: int = 10,
        enable_retry: bool = False,
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=request_timeout
        )
        self.model = model
        self.max_duration = max_duration
        self.request_timeout = request_timeout
        self.max_concurrent = max_concurrent
        self.enable_retry = enable_retry
        self.csv_file: Optional[TextIO] = None
        self.csv_writer: Optional[Any] = None
        self.summary_filename: Optional[Path] = None

    def load_trace(self, trace_file: str = "trace.jsonl") -> List[TraceRequest]:
        """Load trace data from JSONL file with streaming processing"""
        requests = []
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                line_num = 0
                start_timestamp = None
                
                for line in f:  # Stream line by line instead of reading all at once
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Skip metadata line (contains trace_version, created_at, etc)
                        if "trace_version" in data or "created_at" in data:
                            logger.info(f"Found trace metadata: version={data.get('trace_version', 'unknown')}")
                            continue
                        
                        # Parse request data
                        timestamp = float(data["timestamp"])
                        
                        # Track start timestamp for duration filtering
                        if start_timestamp is None:
                            start_timestamp = timestamp
                        
                        # Filter by max_duration early to save memory
                        if self.max_duration > 0 and (timestamp - start_timestamp) > self.max_duration:
                            break
                        
                        # Only support Chat Completions for MVP
                        url = data["url"]
                        if not url.endswith("/chat/completions"):
                            logger.debug(f"Skipping unsupported endpoint: {url}")
                            continue
                        
                        request = TraceRequest(
                            timestamp=timestamp,
                            method=data["method"],
                            url=url,
                            body=data["body"],
                            request_id=data["request_id"]
                        )
                        requests.append(request)

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Skipping malformed line {line_num}: {e}")
                        continue

            # Sort by timestamp (should already be sorted in most cases)
            requests.sort(key=lambda x: x.timestamp)
            
            # Normalize timestamps to start from 0
            if requests and start_timestamp is not None:
                for req in requests:
                    req.timestamp -= start_timestamp

            logger.info(f"Loaded {len(requests)} requests from {trace_file}")
            return requests

        except FileNotFoundError:
            logger.error(f"Trace file {trace_file} not found")
            return []

    def init_csv(self, filename: str = "summary.csv"):
        """Initialize CSV file for real-time writing"""
        self.csv_file = open(filename, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "request_id",
                "ttft",
                "input_token_len",
                "output_token_len",
                "launch_time",
                "finish_time",
                "is_success",
            ]
        )
        self.csv_file.flush()

    def write_result_to_csv(self, result: RequestResult):
        """Write result to CSV immediately"""
        if self.csv_writer and self.csv_file:
            self.csv_writer.writerow(
                [
                    result.request_id,
                    f"{result.ttft:.4f}",
                    result.input_token_len,
                    result.output_token_len,
                    f"{result.launch_time:.4f}",
                    f"{result.finish_time:.4f}",
                    result.is_success,
                ]
            )
            self.csv_file.flush()

    def close_csv(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()

    async def send_request(self, request: TraceRequest) -> RequestResult:
        """Send a single request and measure metrics"""
        launch_time = time.monotonic()
        first_token_time = None
        response_content = ""
        input_tokens = 0
        output_tokens = 0
        max_retries = 3 if self.enable_retry else 1

        for attempt in range(max_retries):
            try:
                # Use model_dump(exclude_unset=True) for optimal parameter handling
                if hasattr(request.body, 'model_dump'):
                    body_dict = request.body.model_dump(exclude_unset=True)
                elif hasattr(request.body, 'dict'):
                    # Compatibility with older pydantic versions
                    body_dict = request.body.dict(exclude_unset=True)
                else:
                    body_dict = request.body.copy() if isinstance(request.body, dict) else request.body

                # Override model if not specified or use our configured model
                if 'model' not in body_dict or not body_dict['model']:
                    body_dict['model'] = self.model

                # Ensure streaming for TTFT measurement
                body_dict['stream'] = True
                # Remove conflicting parameters
                body_dict.pop('stream_options', None)
                
                # Only support Chat Completions endpoint
                if not request.url.endswith("/chat/completions"):
                    raise ValueError(f"Unsupported endpoint: {request.url}")
                
                response = await self.client.chat.completions.create(
                    **body_dict,
                    stream_options={"include_usage": True}
                )
                
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        if first_token_time is None and content.strip():
                            first_token_time = time.monotonic()
                        response_content += content
                    
                    # Get token counts from final chunk with usage info
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

                finish_time = time.monotonic()
                ttft = (first_token_time - launch_time) if first_token_time else 0.0

                result = RequestResult(
                    request_id=request.request_id,
                    ttft=ttft,
                    input_token_len=int(input_tokens),
                    output_token_len=int(output_tokens),
                    launch_time=launch_time,
                    finish_time=finish_time,
                    is_success=True,
                )

                # Write to CSV immediately
                self.write_result_to_csv(result)
                return result

            except asyncio.TimeoutError:
                logger.warning(f"Request {request.request_id} timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error for {request.request_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)  # Wait longer for rate limits
                continue
            except openai.APIConnectionError as e:
                logger.warning(f"Connection error for {request.request_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error(f"Request {request.request_id} failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue

        # All retries failed
        finish_time = time.monotonic()
        result = RequestResult(
            request_id=request.request_id,
            ttft=0.0,
            input_token_len=0,
            output_token_len=0,
            launch_time=launch_time,
            finish_time=finish_time,
            is_success=False,
        )
        self.write_result_to_csv(result)
        return result

    async def replay_trace(self, requests: List[TraceRequest], summary_dir: str = ".", output_filename: str = None):
        """Replay trace requests maintaining relative timing"""
        if not requests:
            logger.warning("No requests to replay")
            return

        # Create summary directory if it doesn't exist
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        filename = output_filename or OUTPUT_FILENAME
        self.summary_filename = Path(summary_dir) / filename

        # Initialize CSV for real-time writing
        self.init_csv(str(self.summary_filename))

        start_time = time.monotonic()
        base_timestamp = requests[0].timestamp

        logger.info(
            f"Starting trace replay with {len(requests)} requests "
            f"over {self.max_duration}s (max concurrent: {self.max_concurrent})"
        )

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def send_with_semaphore(req):
            async with semaphore:
                return await self.send_request(req)

        # Launch all requests at their scheduled times
        tasks = []
        try:
            for i, request in enumerate(requests):
                # Calculate relative delay from trace start
                relative_delay = request.timestamp - base_timestamp
                
                # Wait until it's time to send this request
                target_time = start_time + relative_delay
                current_time = time.monotonic()
                if target_time > current_time:
                    await asyncio.sleep(target_time - current_time)

                # Send the request asynchronously (don't wait for completion)
                task = asyncio.create_task(send_with_semaphore(request))
                tasks.append(task)

                logger.info(
                    f"Launched request {request.request_id} at "
                    f"{relative_delay:.2f}s (endpoint: {request.url}) "
                    f"[{i+1}/{len(requests)}]"
                )

            logger.info("All requests launched, waiting for completion...")

            # Wait for all tasks to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.max_duration + 60  # Extra time for completion
                )
            except asyncio.TimeoutError:
                logger.warning("Some requests did not complete within timeout")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, cancelling remaining requests...")
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait a bit for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some requests did not cancel gracefully")
        
        finally:
            self.close_csv()
            logger.info(f"Trace replay finished. Results saved to {self.summary_filename}")
            
    def print_summary(self):
        """Print performance summary from CSV data"""
        if not self.summary_filename:
            logger.error("Summary filename not set. Cannot print summary.")
            return
        try:
            # Third Party
            import pandas as pd

            df = pd.read_csv(self.summary_filename)

            if len(df) == 0:
                logger.warning("No completed requests to summarize")
                return

            successful_requests = df[df["is_success"]]
            if len(successful_requests) == 0:
                logger.warning("No successful requests to summarize")
                return

            total_requests = len(df)
            success_rate = len(successful_requests) / total_requests
            avg_ttft = successful_requests["ttft"].mean()
            total_input_tokens = successful_requests["input_token_len"].sum()
            total_output_tokens = successful_requests["output_token_len"].sum()

            start_time = successful_requests["launch_time"].min()
            end_time = successful_requests["finish_time"].max()
            total_duration = end_time - start_time

            print("\n" + "=" * 60)
            print("TRACE REPLAY SUMMARY")
            print("=" * 60)
            print(f"Total Requests: {total_requests}")
            print(
                f"Successful Requests: {len(successful_requests)} ({success_rate:.2%})"
            )
            print(f"Total Duration: {total_duration:.2f}s")
            print(f"Average TTFT: {avg_ttft:.4f}s")
            print(f"Total Input Tokens: {total_input_tokens}")
            print(f"Total Output Tokens: {total_output_tokens}")
            if total_duration > 0:
                print(
                    f"Throughput: {len(successful_requests) / total_duration:.2f} req/s"
                )
                print(
                    f"Token Throughput: "
                    f"{(total_output_tokens) / total_duration:.2f} tokens/s"
                )
            print("=" * 60)

        except ImportError:
            logger.warning("pandas not available, skipping summary")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


def setup_arg_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(description="LMCache Benchmark Tool")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--trace_file", type=str, help="Override trace file from config"
    )
    parser.add_argument(
        "--max_duration", type=float, help="Override max duration from config"
    )
    parser.add_argument(
        "--output", type=str, help="Override output file from config"
    )
    return parser


async def main():
    """Main function to run the benchmark tool"""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load configuration from YAML file
    config: dict = {}
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(
            f"Config file {args.config} not found. "
            "Please create it or specify the correct path."
        )
        return
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}")
        return

    # Directory configuration (simplified)
    trace_dir = "."
    summary_dir = "."

    logger.info("Mode: Replay Trace")
    server_config = config.get("server", {})
    replay_config = config.get("replay_config", {})
    advanced_config = config.get("advanced", {})

    # Apply command line overrides
    trace_file_name = args.trace_file or replay_config.get("trace_file", "trace.jsonl")
    max_duration = args.max_duration or replay_config.get("max_duration", 60.0)
    output_filename = args.output or replay_config.get("output_file", "summary.csv")

    # Create and configure the replayer
    default_model = "meta-llama/Llama-3.1-8B-Instruct"
    replayer = TraceReplayer(
        base_url=server_config.get("base_url", "http://localhost:8000/v1"),
        api_key=server_config.get("api_key", "EMPTY"),
        model=server_config.get("model", default_model),
        max_duration=max_duration,
        request_timeout=advanced_config.get("request_timeout", 30.0),
        max_concurrent=advanced_config.get("max_concurrent", 10),
        enable_retry=advanced_config.get("enable_retry", False),
    )

    # Load trace data
    trace_file = Path(trace_dir) / trace_file_name
    requests = replayer.load_trace(trace_file)

    if not requests:
        logger.error(f"No requests loaded from trace file '{trace_file}'. Exiting.")
        return

    # Replay the trace (use custom output filename if specified)
    await replayer.replay_trace(requests, summary_dir=summary_dir, output_filename=output_filename)

    # Print summary
    replayer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
