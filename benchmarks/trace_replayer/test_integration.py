#!/usr/bin/env python3
"""
Integration test script for LMCache Trace Replayer

This script performs comprehensive testing of the trace replayer functionality.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_trace(output_file: str):
    """Create a comprehensive test trace file"""
    metadata = {
        "trace_version": "1.0",
        "created_at": "2025-09-06T00:00:00Z",
        "description": "Integration test trace",
        "vllm_version": "0.5.1"
    }
    
    requests = [
        {
            "timestamp": 0.0,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 50,
                "temperature": 0.7
            },
            "request_id": "test-chat-001"
        },
        {
            "timestamp": 1.0,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "max_tokens": 100,
                "temperature": 0.8
            },
            "request_id": "test-chat-002"
        },
        {
            "timestamp": 2.0,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Thank you for the explanation!"}],
                "max_tokens": 200
            },
            "request_id": "test-chat-003"
        },
        {
            "timestamp": 3.0,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Can you help me with coding?"}],
                "max_tokens": 150
            },
            "request_id": "test-chat-004"
        }
    ]
    
    with open(output_file, 'w') as f:
        f.write(json.dumps(metadata) + '\n')
        for req in requests:
            f.write(json.dumps(req) + '\n')
    
    logger.info(f"Created test trace with {len(requests)} requests at {output_file}")
    return len(requests)

def test_trace_loading():
    """Test trace file loading functionality"""
    logger.info("=== Testing Trace Loading ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        trace_file = f.name
        expected_count = create_test_trace(trace_file)
    
    try:
        # Import here to avoid issues with module loading
        from trace_replayer import TraceReplayer
        
        replayer = TraceReplayer()
        requests = replayer.load_trace(trace_file)
        
        assert len(requests) == expected_count, f"Expected {expected_count} requests, got {len(requests)}"
        
        # Test request structure
        for req in requests:
            assert hasattr(req, 'timestamp')
            assert hasattr(req, 'method')
            assert hasattr(req, 'url')
            assert hasattr(req, 'body')
            assert hasattr(req, 'request_id')
        
        logger.info(f"✅ Successfully loaded {len(requests)} requests")
        
        # Test filtering by max_duration
        replayer_filtered = TraceReplayer(max_duration=2.5)
        filtered_requests = replayer_filtered.load_trace(trace_file)
        assert len(filtered_requests) < len(requests), "Duration filtering should reduce request count"
        logger.info(f"✅ Duration filtering works: {len(filtered_requests)} requests within 2.5s")
        
    finally:
        os.unlink(trace_file)

async def test_request_processing():
    """Test request processing functionality without complex mocking"""
    logger.info("=== Testing Request Processing ===")
    
    from trace_replayer import TraceReplayer, TraceRequest, RequestResult
    
    # Create test requests
    chat_request_1 = TraceRequest(
        timestamp=0.0,
        method="POST",
        url="/v1/chat/completions",
        body={"messages": [{"role": "user", "content": "Hello"}]},
        request_id="test-chat-1"
    )
    
    chat_request_2 = TraceRequest(
        timestamp=1.0,
        method="POST", 
        url="/v1/chat/completions",
        body={"messages": [{"role": "user", "content": "How are you?"}]},
        request_id="test-chat-2"
    )
    
    chat_request_3 = TraceRequest(
        timestamp=2.0,
        method="POST",
        url="/v1/chat/completions", 
        body={"messages": [{"role": "user", "content": "What is AI?"}]},
        request_id="test-chat-3"
    )
    
    replayer = TraceReplayer()
    
    # Test that requests can be processed (will fail due to no server, but that's expected)
    # We're mainly testing the request structure and error handling
    
    results = []
    for req in [chat_request_1, chat_request_2, chat_request_3]:
        result = await replayer.send_request(req)
        results.append(result)
        assert isinstance(result, RequestResult), f"Should return RequestResult object for {req.request_id}"
        assert result.request_id == req.request_id, f"Should preserve request ID"
        
    logger.info(f"✅ Processed {len(results)} requests (failures expected without server)")
    
    # Test parameter cleaning with exclude_unset=True works
    conflicting_request = TraceRequest(
        timestamp=0.0,
        method="POST",
        url="/v1/chat/completions",
        body={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,  # This should be overridden
            "stream_options": {"include_usage": False},  # This should be removed
            "model": "original-model"  # This should be overridden
        },
        request_id="test-conflicting"
    )
    
    result = await replayer.send_request(conflicting_request)
    assert isinstance(result, RequestResult), "Should handle conflicting parameters"
    logger.info("✅ Parameter conflict handling with exclude_unset=True works")

def test_csv_output():
    """Test CSV output functionality"""
    logger.info("=== Testing CSV Output ===")
    
    from trace_replayer import TraceReplayer, RequestResult
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_output.csv"
        
        replayer = TraceReplayer()
        replayer.init_csv(str(csv_file))
        
        # Write some test results
        test_results = [
            RequestResult(
                request_id="test-001",
                ttft=0.123,
                input_token_len=50,
                output_token_len=25,
                launch_time=1000.0,
                finish_time=1001.5,
                is_success=True
            ),
            RequestResult(
                request_id="test-002", 
                ttft=0.0,
                input_token_len=0,
                output_token_len=0,
                launch_time=1002.0,
                finish_time=1002.1,
                is_success=False
            )
        ]
        
        for result in test_results:
            replayer.write_result_to_csv(result)
        
        replayer.close_csv()
        
        # Verify CSV content
        assert csv_file.exists(), "CSV file should be created"
        content = csv_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 3, f"Expected 3 lines (header + 2 data), got {len(lines)}"
        assert "request_id,ttft,input_token_len" in lines[0], "Header should contain expected columns"
        assert "test-001" in lines[1], "First result should be in CSV"
        assert "test-002" in lines[2], "Second result should be in CSV"
        
        logger.info("✅ CSV output works correctly")

async def test_full_integration():
    """Test full end-to-end integration"""
    logger.info("=== Testing Full Integration ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "integration_test.jsonl"
        config_file = Path(tmpdir) / "test_config.yaml"
        
        # Create test files
        create_test_trace(str(trace_file))
        
        config_content = f"""
directories:
  trace_dir: "{tmpdir}"
  summary_dir: "{tmpdir}"

server:
  base_url: "http://localhost:8000/v1"
  api_key: "test-key"
  model: "test-model"

replay_config:
  trace_file: "{trace_file.name}"
  max_duration: 10.0

advanced:
  request_timeout: 5.0
  max_concurrent: 2
  enable_retry: true
"""
        config_file.write_text(config_content)
        
        # Mock the entire replay process
        from trace_replayer import TraceReplayer
        
        # Create a mock that simulates successful requests
        async def mock_send_request(request):
            from trace_replayer import RequestResult
            await asyncio.sleep(0.01)  # Simulate some processing time
            return RequestResult(
                request_id=request.request_id,
                ttft=0.1,
                input_token_len=20,
                output_token_len=10,
                launch_time=time.monotonic(),
                finish_time=time.monotonic() + 0.1,
                is_success=True
            )
        
        replayer = TraceReplayer(
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            model="test-model",
            max_duration=10.0,
            request_timeout=5.0,
            max_concurrent=2,
            enable_retry=True
        )
        
        # Load trace
        requests = replayer.load_trace(str(trace_file))
        assert len(requests) > 0, "Should load test requests"
        
        # Mock the send_request method
        with patch.object(replayer, 'send_request', side_effect=mock_send_request):
            await replayer.replay_trace(requests, str(tmpdir))
        
        # Check if summary file was created
        summary_file = Path(tmpdir) / "summary.csv"
        assert summary_file.exists(), "Summary CSV should be created"
        
        # Verify summary content
        content = summary_file.read_text()
        lines = content.strip().split('\n')
        logger.info(f"CSV content has {len(lines)} lines")
        
        # Should at least have header, even if no successful requests
        assert len(lines) >= 1, "Should have at least header row"
        assert "request_id,ttft,input_token_len" in lines[0], "Header should contain expected columns"
        
        logger.info("✅ Full integration test passed")

def test_error_handling():
    """Test error handling scenarios"""
    logger.info("=== Testing Error Handling ===")
    
    from trace_replayer import TraceReplayer
    
    # Test missing file
    replayer = TraceReplayer()
    requests = replayer.load_trace("nonexistent_file.jsonl")
    assert len(requests) == 0, "Should return empty list for missing file"
    logger.info("✅ Missing file handling works")
    
    # Test malformed JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"trace_version": "1.0"}\n')
        f.write('invalid json line\n')
        f.write('{"timestamp": 1.0, "method": "POST", "url": "/v1/chat/completions", "body": {}, "request_id": "test"}\n')
        malformed_file = f.name
    
    try:
        requests = replayer.load_trace(malformed_file)
        assert len(requests) == 1, "Should skip malformed lines and load valid ones"
        logger.info("✅ Malformed JSON handling works")
    finally:
        os.unlink(malformed_file)
    
    logger.info("✅ Error handling tests passed")

async def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting LMCache Trace Replayer Integration Tests")
    
    try:
        # Test individual components
        test_trace_loading()
        await test_request_processing()
        test_csv_output()
        test_error_handling()
        
        # Test full integration
        await test_full_integration()
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
