import json
import asyncio
import csv
import time
import datetime
import logging
import argparse
import yaml
import random
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_FILENAME = f"summary-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

@dataclass
class TraceRequest:
    chat_id: str
    timestamp: float
    input_text: str
    target_output_length: int

@dataclass
class RequestResult:
    chat_id: str
    ttft: float
    input_token_len: int
    output_token_len: int
    launch_time: float
    finish_time: float
    is_success: bool

class TraceReplayer:
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", model: str = "meta-llama/Llama-3.1-8B-Instruct", max_duration: float = 100.0):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_duration = max_duration
        self.csv_file = None
        self.csv_writer = None
        self.summary_filename = None
        
    def load_trace(self, trace_file: str = "trace.jsonl") -> List[TraceRequest]:
        """Load trace data from JSONL file"""
        requests = []
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            
                            chat_id = str(data['chat_id'])
                            timestamp = float(data['timestamp'])
                            input_text = data['input_text']
                            target_output_length = int(data['output_length'])
                            
                            requests.append(TraceRequest(
                                chat_id=chat_id,
                                timestamp=timestamp,
                                input_text=input_text,
                                target_output_length=target_output_length
                            ))
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping malformed line {line_num}: {e}")
                            continue
            
            # Sort by timestamp and filter by max_duration
            requests.sort(key=lambda x: x.timestamp)
            
            # Normalize timestamps to start from 0
            if requests:
                min_timestamp = requests[0].timestamp
                for req in requests:
                    req.timestamp -= min_timestamp
                
                # Filter requests within max_duration
                requests = [req for req in requests if req.timestamp <= self.max_duration]
            
            logger.info(f"Loaded {len(requests)} requests from {trace_file} (duration: {self.max_duration}s)")
            return requests
            
        except FileNotFoundError:
            logger.error(f"Trace file {trace_file} not found")
            return []
    
    def init_csv(self, filename: str = "summary.csv"):
        """Initialize CSV file for real-time writing"""
        self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['chat_id', 'ttft', 'input_token_len', 'output_token_len', 'launch_time', 'finish_time', 'is_success'])
        self.csv_file.flush()
        
    def write_result_to_csv(self, result: RequestResult):
        """Write result to CSV immediately"""
        if self.csv_writer:
            self.csv_writer.writerow([
                result.chat_id,
                f"{result.ttft:.4f}",
                result.input_token_len,
                result.output_token_len,
                f"{result.launch_time:.4f}",
                f"{result.finish_time:.4f}",
                result.is_success,
            ])
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
        
        try:
            # Build messages
            messages = [{"role": "user", "content": request.input_text}]
            
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0,
                stream=True,
                max_tokens=request.target_output_length,
                stream_options={"include_usage": True}
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if first_token_time is None and content.strip():
                        first_token_time = time.monotonic()
                    response_content += content
            
            finish_time = time.monotonic()
            
            # Get token counts from the last chunk
            input_tokens = chunk.usage.prompt_tokens if hasattr(chunk, 'usage') and chunk.usage else 0
            output_tokens = chunk.usage.completion_tokens if hasattr(chunk, 'usage') and chunk.usage else 0
            
            ttft = (first_token_time - launch_time) if first_token_time else 0.0
            
            result = RequestResult(
                chat_id=request.chat_id,
                ttft=ttft,
                input_token_len=input_tokens,
                output_token_len=output_tokens,
                launch_time=launch_time,
                finish_time=finish_time,
                is_success=True,
            )
            
            # Write to CSV immediately
            self.write_result_to_csv(result)
            return result
            
        except Exception as e:
            logger.error(f"Request {request.chat_id} failed: {e}")
            finish_time = time.monotonic()
            result = RequestResult(
                chat_id=request.chat_id,
                ttft=0.0,
                input_token_len=0,
                output_token_len=0,
                launch_time=launch_time,
                finish_time=finish_time,
                is_success=False,
            )
            # Write failed result to CSV immediately
            self.write_result_to_csv(result)
            return result
    
    async def replay_trace(self, requests: List[TraceRequest], summary_dir: str = "."):
        """Replay trace requests maintaining relative timing"""
        if not requests:
            logger.warning("No requests to replay")
            return
        
        # Create summary directory if it doesn't exist
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        self.summary_filename = Path(summary_dir) / OUTPUT_FILENAME

        # Initialize CSV for real-time writing
        self.init_csv(self.summary_filename)
        
        start_time = time.monotonic()
        
        logger.info(f"Starting trace replay with {len(requests)} requests over {self.max_duration}s")
        
        # Launch all requests at their scheduled times
        tasks = []
        for request in requests:
            # Wait until it's time to send this request
            absolute_send_time = start_time + request.timestamp
            current_time = time.monotonic()
            if absolute_send_time > current_time:
                await asyncio.sleep(absolute_send_time - current_time)
            
            # Send the request asynchronously (don't wait for completion)
            task = asyncio.create_task(self.send_request(request))
            tasks.append(task)
            
            logger.info(f"Launched request {request.chat_id} at {request.timestamp:.2f}s (target output: {request.target_output_length})")
        
        logger.info("All requests launched, waiting for completion...")
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        self.close_csv()
        logger.info(f"Trace replay finished. Results saved to {self.summary_filename}")

    def print_summary(self):
        """Print performance summary from CSV data"""
        if not self.summary_filename:
            logger.error("Summary filename not set. Cannot print summary.")
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.summary_filename)
            
            if len(df) == 0:
                logger.warning("No completed requests to summarize")
                return
            
            successful_requests = df[df['is_success']]
            if len(successful_requests) == 0:
                logger.warning("No successful requests to summarize")
                return

            total_requests = len(df)
            success_rate = len(successful_requests) / total_requests
            avg_ttft = successful_requests['ttft'].mean()
            total_input_tokens = successful_requests['input_token_len'].sum()
            total_output_tokens = successful_requests['output_token_len'].sum()
            
            start_time = successful_requests['launch_time'].min()
            end_time = successful_requests['finish_time'].max()
            total_duration = end_time - start_time
            
            print("\n" + "="*60)
            print("TRACE REPLAY SUMMARY")
            print("="*60)
            print(f"Total Requests: {total_requests}")
            print(f"Successful Requests: {len(successful_requests)} ({success_rate:.2%})")
            print(f"Total Duration: {total_duration:.2f}s")
            print(f"Average TTFT: {avg_ttft:.4f}s")
            print(f"Total Input Tokens: {total_input_tokens}")
            print(f"Total Output Tokens: {total_output_tokens}")
            if total_duration > 0:
                print(f"Throughput: {len(successful_requests)/total_duration:.2f} req/s")
                print(f"Token Throughput: {(total_output_tokens)/total_duration:.2f} tokens/s")
            print("="*60)
            
        except ImportError:
            logger.warning("pandas not available, skipping summary")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

class TraceDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
    def generate_synthetic_conversation_data(self, 
                                           num_requests: int = 100,
                                           duration: float = 60.0,
                                           output_file: str = "trace.jsonl") -> None:
        """Generate synthetic conversational data similar to ShareGPT"""
        
        # Conversation templates based on common LLM use cases
        conversation_templates = [
            # Code assistance
            ("Explain this Python function:", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr", 150),
            
            # Question answering
            ("What is machine learning and how does it work?", "", 200),
            ("Explain the differences between supervised and unsupervised learning.", "", 180),
            
            # Document analysis (simulating long context)
            ("Summarize this document:", "The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. " * 20, 100),
            
            # Creative writing
            ("Write a short story about a robot learning to paint.", "", 300),
            ("Create a poem about autumn leaves.", "", 120),
            
            # Technical explanation
            ("How do neural networks work? Explain in simple terms.", "", 250),
            ("What are the key principles of database design?", "", 200),
            
            # Multi-turn conversation starters
            ("I'm building a web application and need advice on architecture.", "", 180),
            ("Can you help me debug this code error?", "", 150),
        ]
        
        # Domain-specific long prompts (simulating RAG scenarios)
        long_document_templates = [
            ("Based on this research paper, what are the main conclusions? Paper: ", "Abstract: " + "Deep learning has revolutionized artificial intelligence by enabling machines to learn complex patterns from data. " * 100, 200),
            ("Analyze this legal document and identify key clauses: ", "CONTRACT AGREEMENT: " + "The parties agree to the following terms and conditions. " * 80, 150),
            ("Review this medical case study: ", "Patient History: " + "A 45-year-old patient presented with symptoms including. " * 60, 180),
        ]
        
        requests_data = []
        
        # Generate timestamps with realistic arrival patterns
        # Using exponential distribution for realistic request timing
        timestamps = []
        current_time = 0.0
        for i in range(num_requests):
            # Add some randomness: burst periods and quiet periods
            if random.random() < 0.3:  # 30% chance of burst
                # Shorter intervals for bursts
                mean_interval = duration / (num_requests * 2)
            else:
                mean_interval = duration / num_requests
            
            # Simple exponential distribution using inverse transform
            interval = -mean_interval * math.log(1 - random.random())
            current_time += interval
            if current_time > duration:
                break
            timestamps.append(current_time)
        
        # Generate requests
        for i, timestamp in enumerate(timestamps):
            # Mix short and long prompts (80% short, 20% long)
            if random.random() < 0.8:
                template = random.choice(conversation_templates)
                prompt = template[0]
                if template[1]:  # Add context if available
                    prompt += "\n\n" + template[1]
                target_length = template[2]
            else:
                template = random.choice(long_document_templates)
                prompt = template[0] + template[1]
                target_length = template[2]
            
            # Add some variation to output length
            target_length = max(10, target_length + random.randint(-50, 50))
            
            request = {
                "chat_id": f"chat_{i:04d}",
                "timestamp": timestamp,
                "input_text": prompt,
                "output_length": target_length
            }
            requests_data.append(request)
        
        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests_data:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Generated {len(requests_data)} requests in {output_file}")
        logger.info(f"Duration: {duration:.1f}s, Average rate: {len(requests_data)/duration:.2f} req/s")

    def convert_sharegpt_to_trace(self, 
                                  sharegpt_file: str,
                                  output_file: str = "sharegpt_trace.jsonl",
                                  max_requests: int = 1000,
                                  duration: float = 60.0) -> None:
        """Convert ShareGPT format to trace format"""
        
        try:
            with open(sharegpt_file, 'r', encoding='utf-8') as f:
                sharegpt_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ShareGPT file: {e}")
            return
        
        requests_data = []
        
        # Generate timestamps
        timestamps = sorted([random.uniform(0, duration) for _ in range(min(max_requests, len(sharegpt_data)))])
        
        for i, (conversation, timestamp) in enumerate(zip(sharegpt_data[:max_requests], timestamps)):
            try:
                # Extract the first human message as input
                if 'conversations' in conversation:
                    human_messages = [msg for msg in conversation['conversations'] 
                                    if msg.get('from') == 'human']
                    if human_messages:
                        input_text = human_messages[0]['value']
                        
                        # Estimate output length based on assistant response
                        assistant_messages = [msg for msg in conversation['conversations'] 
                                            if msg.get('from') == 'gpt' or msg.get('from') == 'assistant']
                        if assistant_messages:
                            # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
                            output_length = len(assistant_messages[0]['value']) // 4
                        else:
                            output_length = 100  # default
                        
                        request = {
                            "chat_id": f"sharegpt_{i:04d}",
                            "timestamp": timestamp,
                            "input_text": input_text,
                            "output_length": max(10, min(500, output_length))  # clamp between 10-500
                        }
                        requests_data.append(request)
            except Exception as e:
                logger.warning(f"Skipping conversation {i}: {e}")
                continue
        
        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests_data:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Converted {len(requests_data)} conversations to {output_file}")

    def generate_caching_focused_trace(self,
                                     num_requests: int = 200,
                                     duration: float = 120.0,
                                     cache_hit_ratio: float = 0.6,
                                     output_file: str = "cache_trace.jsonl") -> None:
        """Generate trace data specifically designed to test caching effectiveness"""
        
        # Base documents that will be reused (for cache hits)
        base_documents = []
        for i in range(20):  # 20 different base documents
            doc_content = f"Document {i}: " + " ".join([f"content_word_{j}" for j in range(1000)])
            base_documents.append(doc_content)
        
        # Query templates that can be applied to any document
        query_templates = [
            "Summarize this document:",
            "What are the key points in this text?",
            "Extract the main themes from:",
            "Provide a brief overview of:",
            "Analyze the content of:",
            "What insights can you derive from:",
        ]
        
        requests_data = []
        timestamps = sorted([random.uniform(0, duration) for _ in range(num_requests)])
        
        # Track which documents have been used (for realistic cache behavior)
        used_documents = []
        
        for i, timestamp in enumerate(timestamps):
            # Decide whether this should be a cache hit or miss
            if len(used_documents) > 0 and random.random() < cache_hit_ratio:
                # Cache hit: reuse a previous document
                doc_content = random.choice(used_documents)
                chat_id_suffix = "hit"
            else:
                # Cache miss: use a new document
                doc_content = random.choice(base_documents)
                if doc_content not in used_documents:
                    used_documents.append(doc_content)
                chat_id_suffix = "miss"
            
            # Create the full prompt
            query = random.choice(query_templates)
            full_prompt = f"{query}\n\n{doc_content}"
            
            request = {
                "chat_id": f"cache_{i:04d}_{chat_id_suffix}",
                "timestamp": timestamp,
                "input_text": full_prompt,
                "output_length": random.randint(50, 200)
            }
            requests_data.append(request)
        
        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests_data:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Generated {len(requests_data)} cache-focused requests in {output_file}")
        actual_hits = sum(1 for r in requests_data if "hit" in r["chat_id"])
        logger.info(f"Target cache hit ratio: {cache_hit_ratio:.1%}, Actual: {actual_hits/len(requests_data):.1%}")

def setup_arg_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(description="LMCache Benchmark Tool")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file path")
    return parser

async def main():
    """Main function to run the benchmark tool"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = {}
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Config file {args.config} not found. Please create it or specify the correct path.")
        return
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}")
        return

    # Get the execution mode
    mode = config.get('mode', 'replay') # Default to replay for backward compatibility
    
    # Directory configuration
    dir_config = config.get('directories', {})
    trace_dir = dir_config.get('trace_dir', 'traces')
    summary_dir = dir_config.get('summary_dir', 'summaries')

    if mode == 'generate':
        logger.info("Mode: Generate Trace")
        gen_config = config.get('generate_config', {})
        gen_mode = gen_config.get('generation_mode', 'synthetic')
        
        generator = TraceDataGenerator(seed=gen_config.get('seed', 42))
        
        # Ensure trace directory exists
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(trace_dir) / gen_config.get('output_file', 'trace.jsonl')
        
        if gen_mode == "synthetic":
            generator.generate_synthetic_conversation_data(
                num_requests=gen_config.get('num_requests', 100),
                duration=gen_config.get('duration', 60.0),
                output_file=output_file
            )
        elif gen_mode == "sharegpt":
            if not gen_config.get('input_file'):
                logger.error("'input_file' is required for sharegpt generation mode")
                return
            generator.convert_sharegpt_to_trace(
                sharegpt_file=gen_config.get('input_file'),
                output_file=output_file,
                max_requests=gen_config.get('num_requests', 1000),
                duration=gen_config.get('duration', 60.0)
            )
        elif gen_mode == "cache-focused":
            generator.generate_caching_focused_trace(
                num_requests=gen_config.get('num_requests', 200),
                duration=gen_config.get('duration', 120.0),
                cache_hit_ratio=gen_config.get('cache_hit_ratio', 0.6),
                output_file=output_file
            )
        else:
            logger.error(f"Unknown generation mode: {gen_mode}")

    elif mode == 'replay':
        logger.info("Mode: Replay Trace")
        server_config = config.get('server', {})
        replay_config = config.get('replay_config', {})
        
        # Create and configure the replayer
        replayer = TraceReplayer(
            base_url=server_config.get('base_url', 'http://localhost:8000/v1'),
            api_key=server_config.get('api_key', 'EMPTY'),
            model=server_config.get('model', 'meta-llama/Llama-3.1-8B-Instruct'),
            max_duration=replay_config.get('max_duration', 60.0)
        )
        
        # Load trace data
        trace_file = Path(trace_dir) / replay_config.get('trace_file', 'trace.jsonl')
        requests = replayer.load_trace(trace_file)
        
        if not requests:
            logger.error(f"No requests loaded from trace file '{trace_file}'. Exiting.")
            return
        
        # Replay the trace
        await replayer.replay_trace(requests, summary_dir=summary_dir)
        
        # Print summary
        replayer.print_summary()
        
    else:
        logger.error(f"Invalid mode '{mode}' in config. Use 'generate' or 'replay'.")

if __name__ == "__main__":
    asyncio.run(main())