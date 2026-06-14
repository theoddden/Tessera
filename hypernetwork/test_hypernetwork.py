"""Tests for Tessera Hypernetwork Service"""

import pytest
from hypernetwork.doc_to_lora import DocToLoRA
from hypernetwork.text_to_lora import TextToLoRA
from hypernetwork.metadata_to_lora import MetadataToLoRA


@pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
def test_text_to_lora_initialization():
    """Test TextToLoRA can be initialized"""
    model = TextToLoRA("meta-llama/Llama-3-8B")
    assert model is not None
    assert model.base_model == "meta-llama/Llama-3-8B"


@pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
def test_doc_to_lora_initialization():
    """Test DocToLoRA can be initialized"""
    model = DocToLoRA("meta-llama/Llama-3-8B")
    assert model is not None
    assert model.base_model == "meta-llama/Llama-3-8B"


@pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
def test_metadata_to_lora_initialization():
    """Test MetadataToLoRA can be initialized"""
    model = MetadataToLoRA("meta-llama/Llama-3-8B")
    assert model is not None
    assert model.base_model == "meta-llama/Llama-3-8B"


@pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
def test_text_to_lora_projection_layers():
    """Test that projection layers are initialized"""
    model = TextToLoRA("meta-llama/Llama-3-8B")
    assert model.proj_lora_A is not None
    assert model.proj_lora_B is not None


@pytest.mark.skip(reason="Requires HuggingFace models, skip in CI to avoid rate limiting")
def test_doc_to_lora_shine_processor():
    """Test that SHINE processor is initialized"""
    model = DocToLoRA("meta-llama/Llama-3-8B")
    assert model.shine_processor is not None
