"""Tests for Tessera Hypernetwork Service"""

import time
import requests
import json


# Original model initialization tests - commented out due to environment dependency issues
# @pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
# def test_text_to_lora_initialization():
#     """Test TextToLoRA can be initialized"""
#     from tessera_hypernetwork.text_to_lora import TextToLoRA
#     model = TextToLoRA("meta-llama/Llama-3-8B")
#     assert model is not None
#     assert model.base_model == "meta-llama/Llama-3-8B"
#
#
# @pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
# def test_doc_to_lora_initialization():
#     """Test DocToLoRA can be initialized"""
#     from tessera_hypernetwork.doc_to_lora import DocToLoRA
#     model = DocToLoRA("meta-llama/Llama-3-8B")
#     assert model is not None
#     assert model.base_model == "meta-llama/Llama-3-8B"
#
#
# @pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
# def test_metadata_to_lora_initialization():
#     """Test MetadataToLoRA can be initialized"""
#     from tessera_hypernetwork.metadata_to_lora import MetadataToLoRA
#     model = MetadataToLoRA("meta-llama/Llama-3-8B")
#     assert model is not None
#     assert model.base_model == "meta-llama/Llama-3-8B"
#
#
# @pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
# def test_text_to_lora_projection_layers():
#     """Test that projection layers are initialized"""
#     from tessera_hypernetwork.text_to_lora import TextToLoRA
#     model = TextToLoRA("meta-llama/Llama-3-8B")
#     assert model.proj_lora_A is not None
#     assert model.proj_lora_B is not None
#
#
# @pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
# def test_doc_to_lora_shine_processor():
#     """Test that SHINE processor is initialized"""
#     from tessera_hypernetwork.doc_to_lora import DocToLoRA
#     model = DocToLoRA("meta-llama/Llama-3-8B")
#     assert model.shine_processor is not None


def test_adapter_generation_latency():
    """Test adapter generation latency via /v1/generate endpoint"""
    base_url = "http://localhost:8080"
    base_model = "meta-llama/Llama-3-8B"
    target_rank = 16

    # Test metadata mode
    metadata = {"task": "classification", "domain": "medical"}
    latencies = []

    for i in range(5):
        start = time.time()
        response = requests.post(
            f"{base_url}/v1/generate",
            json={
                "messages": [{"role": "user", "content": json.dumps(metadata)}],
                "base_model": base_model,
                "target_rank": target_rank,
                "response_format": {"type": "safetensors"},
            },
            timeout=120,
        )
        end = time.time()
        latency = end - start
        latencies.append(latency)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > 0, "Empty response"

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print("\nAdapter Generation Latency (metadata mode):")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Min: {min_latency:.3f}s")
    print(f"  Max: {max_latency:.3f}s")

    # Assert average latency is reasonable (< 10 seconds for placeholder)
    assert avg_latency < 10.0, f"Average latency {avg_latency:.3f}s exceeds threshold"


def test_adapter_generation_batch_latency():
    """Test batch adapter generation latency"""
    base_url = "http://localhost:8080"
    base_model = "meta-llama/Llama-3-8B"
    target_rank = 16
    batch_size = 10

    metadata_packets = [
        {"id": f"adapter_{i}", "task": "classification", "domain": f"domain_{i}"}
        for i in range(batch_size)
    ]

    start = time.time()
    successful = 0

    for meta in metadata_packets:
        response = requests.post(
            f"{base_url}/v1/generate",
            json={
                "messages": [{"role": "user", "content": json.dumps(meta)}],
                "base_model": base_model,
                "target_rank": target_rank,
                "response_format": {"type": "safetensors"},
            },
            timeout=120,
        )
        if response.status_code == 200:
            successful += 1

    end = time.time()
    total_time = end - start
    avg_per_adapter = total_time / batch_size

    print(f"\nBatch Adapter Generation ({batch_size} adapters):")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average per adapter: {avg_per_adapter:.3f}s")
    print(f"  Successful: {successful}/{batch_size}")

    assert successful == batch_size, (
        f"Only {successful}/{batch_size} adapters generated successfully"
    )
