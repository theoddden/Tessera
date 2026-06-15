"""
QUICK START: Import metadata and generate adapter FAST.

This is the simplest way to use Tessera Hypernetwork.
"""

from tessera_hypernetwork.metadata_to_lora import MetadataToLoRA


def generate_adapter_simple(metadata: dict, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Generate a LoRA adapter from metadata.

    Args:
        metadata: Dictionary with domain info
        base_model: Base model name

    Returns:
        LoRA weights (lora_A, lora_B)
    """
    # Initialize hypernetwork
    hypernetwork = MetadataToLoRA(base_model=base_model)

    # Generate adapter
    lora_weights = hypernetwork.generate(metadata, rank=16)

    return lora_weights


if __name__ == "__main__":
    # Example: Generate legal adapter
    metadata = {
        "domain": "legal",
        "role": "attorney",
        "specialty": "contract_law",
        "jurisdiction": "US"
    }

    print("Generating adapter...")
    lora = generate_adapter_simple(metadata)

    print("✓ Adapter generated!")
    print(f"  LoRA A shape: {lora['lora_A'].shape}")
    print(f"  LoRA B shape: {lora['lora_B'].shape}")
    print(f"  Ready to use with {metadata['domain']} domain")
