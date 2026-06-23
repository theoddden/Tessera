"""
Advanced training script for MetadataToLoRA hypernetwork.

Implements asymmetric training with:
- Domain-conditioned normalization (MLP instead of linear projection)
- Structured metadata encoding (per-field embeddings)
- Curriculum training by domain difficulty
- Cross-domain evaluation to monitor contamination
- Domain-averaged LoRA initialization
- Embedding similarity checks to detect encoder collapse
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sentence_transformers import SentenceTransformer
import safetensors.torch
from dataclasses import dataclass
import numpy as np


@dataclass
class DomainConfig:
    """Configuration for domain-specific training parameters."""

    name: str
    difficulty: int  # 1-5, higher = harder
    priority: int  # Training order (lower = earlier)
    vocabulary_distinctiveness: float  # 0-1, higher = more distinct vocab


# Domain difficulty ranking based on observations
DOMAIN_CONFIGS = {
    "international_law": DomainConfig(
        "international_law", difficulty=1, priority=1, vocabulary_distinctiveness=0.9
    ),
    "jurisprudence": DomainConfig(
        "jurisprudence", difficulty=1, priority=1, vocabulary_distinctiveness=0.85
    ),
    "medical": DomainConfig(
        "medical", difficulty=2, priority=2, vocabulary_distinctiveness=0.8
    ),
    "computer_science": DomainConfig(
        "computer_science", difficulty=3, priority=3, vocabulary_distinctiveness=0.7
    ),
    "high_school_statistics": DomainConfig(
        "high_school_statistics",
        difficulty=5,
        priority=5,
        vocabulary_distinctiveness=0.4,
    ),
    "econometrics": DomainConfig(
        "econometrics", difficulty=5, priority=5, vocabulary_distinctiveness=0.3
    ),
}


class StructuredMetadataEncoder(nn.Module):
    """Encodes metadata with per-field embeddings for richer representation."""

    def __init__(self, base_encoder: SentenceTransformer, embed_dim: int = 768):
        super().__init__()
        self.base_encoder = base_encoder
        self.embed_dim = embed_dim

        # Field-specific projection layers
        self.domain_proj = nn.Linear(embed_dim, embed_dim)
        self.role_proj = nn.Linear(embed_dim, embed_dim)
        self.specialty_proj = nn.Linear(embed_dim, embed_dim)
        self.jurisdiction_proj = nn.Linear(embed_dim, embed_dim)

        # Fusion layer to combine field embeddings
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=2,
        )

    def forward(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Encode structured metadata into single embedding."""
        device = next(self.parameters()).device

        # Encode each field separately
        fields = []
        field_values = []

        for field_name, proj in [
            ("domain", self.domain_proj),
            ("role", self.role_proj),
            ("specialty", self.specialty_proj),
            ("jurisdiction", self.jurisdiction_proj),
        ]:
            value = metadata.get(field_name, "")
            if value:
                text = str(value)
                with torch.no_grad():
                    embedding = self.base_encoder.encode(
                        text, convert_to_tensor=True, show_progress_bar=False
                    ).to(device)
                projected = proj(embedding)
                fields.append(projected.unsqueeze(0))  # Add sequence dimension
                field_values.append(field_name)

        if not fields:
            # Fallback: encode full metadata and pass through fusion to maintain consistent shape
            metadata_text = json.dumps(metadata, indent=2)
            with torch.no_grad():
                embedding = self.base_encoder.encode(
                    metadata_text, convert_to_tensor=True, show_progress_bar=False
                ).to(device)
            # fusion expects (batch, seq, embed_dim) with batch_first=True
            fused = self.fusion(
                embedding.unsqueeze(0).unsqueeze(0)
            )  # (1, 1, embed_dim)
            return fused.squeeze(0).squeeze(0)  # (embed_dim,)

        # Stack and fuse field embeddings
        field_embeddings = torch.cat(fields, dim=0)  # (num_fields, embed_dim)
        # fusion expects (batch, seq, embed_dim) — add batch dim
        fused = self.fusion(field_embeddings.unsqueeze(0)).squeeze(
            0
        )  # (num_fields, embed_dim)

        # Aggregate (mean pooling)
        aggregated = fused.mean(dim=0)
        return aggregated


class DomainConditionedHypernetwork(nn.Module):
    """Advanced hypernetwork with domain-conditioned normalization."""

    def __init__(
        self,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out

        # Domain embedding layer
        self.domain_embedding = nn.Embedding(num_domains, embed_dim)

        # MLP projection instead of linear
        self.mlp_lora_A = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # Concatenate with domain embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, rank * d_in),
        )

        self.mlp_lora_B = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_out * rank),
        )

        # Domain-specific scaling factors
        self.domain_scale_A = nn.Parameter(torch.ones(num_domains))
        self.domain_scale_B = nn.Parameter(torch.ones(num_domains))

    def forward(
        self, metadata_embedding: torch.Tensor, domain_id: int
    ) -> Dict[str, torch.Tensor]:
        """Generate LoRA weights conditioned on domain."""
        device = metadata_embedding.device

        # Get domain embedding
        domain_emb = self.domain_embedding(torch.tensor([domain_id], device=device))

        # Ensure both tensors have same number of dimensions
        # metadata_embedding is [embed_dim] (1D), domain_emb is [1, embed_dim] (2D)
        domain_emb = domain_emb.squeeze(0)

        # Concatenate metadata and domain embeddings
        combined = torch.cat([metadata_embedding, domain_emb], dim=-1)

        # Generate LoRA weights through MLP
        lora_A_flat = self.mlp_lora_A(combined)
        lora_B_flat = self.mlp_lora_B(combined)

        # Reshape and apply domain-specific scaling
        lora_A = lora_A_flat.view(-1, self.rank, self.d_in)
        lora_B = lora_B_flat.view(-1, self.d_out, self.rank)

        lora_A = lora_A * self.domain_scale_A[domain_id]
        lora_B = lora_B * self.domain_scale_B[domain_id]

        return {
            "lora_A": lora_A.squeeze(0),
            "lora_B": lora_B.squeeze(0),
        }


class LoRATargetDataset(Dataset):
    """Dataset of (metadata, target_LoRA) pairs for hypernetwork training."""

    def __init__(self, metadata_dir: str, targets_dir: str):
        """
        Args:
            metadata_dir: Directory containing JSON metadata files
            targets_dir: Directory containing target LoRA safetensors files
        """
        self.metadata_dir = Path(metadata_dir)
        self.targets_dir = Path(targets_dir)
        self.samples = self._load_samples()
        self.domain_to_id = self._build_domain_mapping()

    def _build_domain_mapping(self) -> Dict[str, int]:
        """Build mapping from domain names to IDs."""
        domains = set()
        for metadata, _ in self.samples:
            domain = metadata.get("domain", "general")
            domains.add(domain)
        return {domain: idx for idx, domain in enumerate(sorted(domains))}

    def _load_samples(self) -> List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
        """Load metadata and target LoRA pairs with robust error handling."""
        samples = []

        print(f"Loading metadata from: {self.metadata_dir}")
        print(f"Loading targets from: {self.targets_dir}")

        # Load all metadata files
        metadata_files = sorted(self.metadata_dir.glob("*.json"))
        print(f"Found {len(metadata_files)} JSON files in metadata directory")

        if len(metadata_files) == 0:
            print(f"WARNING: No JSON files found in {self.metadata_dir}")
            return []

        for meta_file in metadata_files:
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)

                # Handle various JSON structures
                metadata_list = []
                if isinstance(metadata, dict):
                    metadata_list = [metadata]
                elif isinstance(metadata, list):
                    metadata_list = [m for m in metadata if isinstance(m, dict)]
                else:
                    print(
                        f"Warning: Unexpected JSON structure in {meta_file.name}, skipping"
                    )
                    continue

                for meta in metadata_list:
                    adapter_id = meta.get("id", meta_file.stem)
                    target_file = self.targets_dir / f"{adapter_id}.safetensors"

                    if target_file.exists():
                        # Load target LoRA weights
                        target_weights = safetensors.torch.load_file(str(target_file))
                        samples.append((meta, target_weights))
                    else:
                        print(f"Warning: No target file for {adapter_id}, skipping")
            except json.JSONDecodeError as e:
                print(f"Error parsing {meta_file.name}: {e}")
            except Exception as e:
                print(f"Error loading {meta_file.name}: {e}")

        print(f"Loaded {len(samples)} training samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        metadata, target_weights = self.samples[idx]
        domain_id = self.domain_to_id.get(metadata.get("domain", "general"), 0)
        return metadata, target_weights, domain_id


class SyntheticTargetDataset(Dataset):
    """Synthetic dataset for initial hypernetwork training.

    Generates domain-specific target LoRA weights using domain-specific
    random seeds to create asymmetric but deterministic targets.
    """

    def __init__(self, metadata_dir: str, base_model: str, rank: int = 16):
        """
        Args:
            metadata_dir: Directory containing JSON metadata files
            base_model: Base model identifier
            rank: LoRA rank
        """
        self.metadata_dir = Path(metadata_dir)
        self.base_model = base_model
        self.rank = rank
        self.samples = self._load_metadata()
        self.domain_to_id = self._build_domain_mapping()

    def _build_domain_mapping(self) -> Dict[str, int]:
        """Build mapping from domain names to IDs."""
        domains = set()
        for metadata in self.samples:
            domain = metadata.get("domain", "general")
            domains.add(domain)
        return {domain: idx for idx, domain in enumerate(sorted(domains))}

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load all metadata files with robust error handling."""
        metadata_dir = Path(self.metadata_dir)
        print(f"Loading metadata from: {metadata_dir}")
        print(f"Directory exists: {metadata_dir.exists()}")

        if not metadata_dir.exists():
            print(f"ERROR: Metadata directory does not exist: {metadata_dir}")
            return []

        metadata_files = sorted(metadata_dir.glob("*.json"))
        print(f"Found {len(metadata_files)} JSON files")

        if len(metadata_files) == 0:
            print(f"WARNING: No JSON files found in {metadata_dir}")
            # Try listing all files for debugging
            all_files = list(metadata_dir.iterdir()) if metadata_dir.exists() else []
            print(f"All files in directory: {[f.name for f in all_files]}")
            return []

        samples = []
        for meta_file in metadata_files:
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)

                # Handle various JSON structures
                if isinstance(metadata, dict):
                    # Single metadata object
                    if "domain" not in metadata:
                        print(
                            f"Warning: {meta_file.name} missing 'domain' field, using 'general'"
                        )
                        metadata["domain"] = "general"
                    samples.append(metadata)
                elif isinstance(metadata, list):
                    # List of metadata objects
                    for item in metadata:
                        if isinstance(item, dict):
                            if "domain" not in item:
                                print(
                                    f"Warning: Item in {meta_file.name} missing 'domain' field, using 'general'"
                                )
                                item["domain"] = "general"
                            samples.append(item)
                        else:
                            print(
                                f"Warning: Skipping non-dict item in {meta_file.name}"
                            )
                else:
                    print(
                        f"Warning: Unexpected JSON structure in {meta_file.name}, skipping"
                    )
            except json.JSONDecodeError as e:
                print(f"Error parsing {meta_file.name}: {e}")
            except Exception as e:
                print(f"Error loading {meta_file.name}: {e}")

        print(f"Loaded {len(samples)} metadata samples")
        return samples

    def _get_model_dimensions(self) -> Tuple[int, int]:
        """Get model dimensions."""
        model_dims = {
            "meta-llama/Llama-3-8B": (4096, 4096),
            "meta-llama/Llama-3-70B": (8192, 8192),
            "Qwen/Qwen2-7B": (3584, 3584),
            "deepseek-ai/DeepSeek-V3": (7168, 7168),
            "mistralai/Mistral-7B-Instruct-v0.2": (4096, 4096),
        }
        return model_dims.get(self.base_model, (4096, 4096))

    def _generate_target_weights(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic target LoRA weights based on domain.

        Uses domain-specific random seeds to create deterministic but
        domain-differentiated weight targets.
        """
        domain = metadata.get("domain", "general")
        # Use domain name as seed for reproducibility via a local Generator
        # (avoids corrupting the global RNG state during DataLoader iteration)
        seed = hash(domain) % (2**32)
        rng = torch.Generator()
        rng.manual_seed(seed)

        d_in, d_out = self._get_model_dimensions()

        # Generate domain-specific weights with vocabulary distinctiveness factor
        vocab_distinctiveness = DOMAIN_CONFIGS.get(
            domain,
            DomainConfig(
                domain, difficulty=3, priority=3, vocabulary_distinctiveness=0.5
            ),
        ).vocabulary_distinctiveness

        # Higher distinctiveness = larger magnitude weights
        scale = 0.1 * (1.0 + vocab_distinctiveness)
        lora_A = torch.randn(self.rank, d_in, generator=rng) * scale
        lora_B = torch.randn(d_out, self.rank, generator=rng) * scale

        return {
            "lora_A": lora_A,
            "lora_B": lora_B,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        metadata = self.samples[idx]
        target_weights = self._generate_target_weights(metadata)
        domain_id = self.domain_to_id.get(metadata.get("domain", "general"), 0)
        return metadata, target_weights, domain_id


class EmbeddingSimilarityChecker:
    """Checks for encoder collapse by monitoring embedding similarity."""

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.similarity_history = []

    def check_similarity(
        self, metadata_list: List[Dict[str, Any]], threshold: float = 0.95
    ) -> Dict[str, float]:
        """Check average pairwise cosine similarity of metadata embeddings."""
        embeddings = []
        for metadata in metadata_list:
            text = json.dumps(metadata, indent=2)
            emb = self.encoder.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute pairwise similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t())

        # Get upper triangle (excluding diagonal)
        n = len(embeddings)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        similarities = similarity_matrix[mask]

        avg_similarity = similarities.mean().item()
        max_similarity = similarities.max().item()

        self.similarity_history.append(avg_similarity)

        result = {
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "collapse_detected": avg_similarity > threshold,
        }

        if result["collapse_detected"]:
            print(
                f"WARNING: Encoder collapse detected! Avg similarity: {avg_similarity:.3f}"
            )

        return result


class CrossDomainEvaluator:
    """Evaluates cross-domain contamination of adapters."""

    def __init__(self, metadata_dir: str):
        self.metadata_dir = Path(metadata_dir)
        self.domain_samples = self._group_by_domain()

    def _group_by_domain(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group metadata samples by domain."""
        groups = {}
        for meta_file in sorted(self.metadata_dir.glob("*.json")):
            with open(meta_file) as f:
                metadata = json.load(f)
            domain = metadata.get("domain", "general")
            if domain not in groups:
                groups[domain] = []
            groups[domain].append(metadata)
        return groups

    def evaluate_contamination(
        self,
        hypernetwork: DomainConditionedHypernetwork,
        encoder: StructuredMetadataEncoder,
        device: str = "cuda",
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate cross-domain contamination by generating adapters for each domain
        and checking if they leak into other domains."""
        results = {}

        for source_domain, source_samples in self.domain_samples.items():
            if not source_samples:
                continue

            # Generate adapter for source domain
            source_meta = source_samples[0]
            source_emb = encoder(source_meta)
            source_domain_id = list(self.domain_samples.keys()).index(source_domain)
            source_adapter = hypernetwork(source_emb, source_domain_id)

            # Evaluate against all other domains
            contamination_scores = {}
            for target_domain in self.domain_samples.keys():
                if target_domain == source_domain:
                    continue

                target_meta = self.domain_samples[target_domain][0]
                target_emb = encoder(target_meta)
                target_domain_id = list(self.domain_samples.keys()).index(target_domain)
                target_adapter = hypernetwork(target_emb, target_domain_id)

                # Compute weight similarity (should be low for different domains)
                sim_a = torch.nn.functional.cosine_similarity(
                    source_adapter["lora_A"].flatten(),
                    target_adapter["lora_A"].flatten(),
                    dim=0,
                ).item()
                sim_b = torch.nn.functional.cosine_similarity(
                    source_adapter["lora_B"].flatten(),
                    target_adapter["lora_B"].flatten(),
                    dim=0,
                ).item()

                contamination_scores[target_domain] = (sim_a + sim_b) / 2

            results[source_domain] = contamination_scores

        return results


def initialize_with_domain_averages(
    hypernetwork: DomainConditionedHypernetwork,
    metadata_dir: str,
    base_model: str,
    rank: int = 16,
):
    """Initialize hypernetwork with domain-averaged LoRA weights.

    Computes average LoRA direction for each domain and uses it as
    initialization target instead of zero.
    """
    metadata_dir = Path(metadata_dir)
    domain_samples = {}

    # Group by domain
    for meta_file in sorted(metadata_dir.glob("*.json")):
        with open(meta_file) as f:
            metadata = json.load(f)
        domain = metadata.get("domain", "general")
        if domain not in domain_samples:
            domain_samples[domain] = []
        domain_samples[domain].append(metadata)

    # For each domain, generate a representative target
    domain_targets = {}
    for domain, samples in domain_samples.items():
        if not samples:
            continue

        # Use domain-specific seed for consistent initialization
        seed = hash(domain) % (2**32)
        torch.manual_seed(seed)

        # Get model dimensions
        model_dims = {
            "meta-llama/Llama-3-8B": (4096, 4096),
            "meta-llama/Llama-3-70B": (8192, 8192),
            "Qwen/Qwen2-7B": (3584, 3584),
            "deepseek-ai/DeepSeek-V3": (7168, 7168),
            "mistralai/Mistral-7B-Instruct-v0.2": (4096, 4096),
        }
        d_in, d_out = model_dims.get(base_model, (4096, 4096))

        # Generate domain-specific initialization
        vocab_distinctiveness = DOMAIN_CONFIGS.get(
            domain,
            DomainConfig(
                domain, difficulty=3, priority=3, vocabulary_distinctiveness=0.5
            ),
        ).vocabulary_distinctiveness
        scale = 0.05 * (1.0 + vocab_distinctiveness)

        domain_targets[domain] = {
            "lora_A": torch.randn(rank, d_in) * scale,
            "lora_B": torch.randn(d_out, rank) * scale,
        }

    print(f"Initialized {len(domain_targets)} domain-specific targets")
    return domain_targets


def safe_metadata_collate(batch):
    """Custom collate function that handles missing keys in metadata.

    Keeps metadata as dicts but stringifies individual values that are None or lists.
    Properly collates target weights as dicts of stacked tensors.
    """
    # Separate metadata, target_weights, and domain_id
    metadata_list = []
    target_weights_list = []
    domain_id_list = []

    # Collect all keys present across all metadata items
    all_keys = set()
    for metadata, _, _ in batch:
        all_keys.update(metadata.keys())

    # Normalize each metadata dict to have all keys with cleaned values
    for metadata, target_weights, domain_id in batch:
        cleaned_metadata = {}
        for key in all_keys:
            value = metadata.get(key, None)
            # Only stringify individual values, keep as dict
            if value is None:
                cleaned_metadata[key] = ""
            elif isinstance(value, list):
                cleaned_metadata[key] = json.dumps(value)
            else:
                cleaned_metadata[key] = str(value) if value is not None else ""
        metadata_list.append(cleaned_metadata)
        target_weights_list.append(target_weights)
        domain_id_list.append(domain_id)

    # Collate target weights as dict of stacked tensors
    collated_targets = {}
    if target_weights_list:
        # Get all keys from target weights (should be lora_A, lora_B)
        target_keys = target_weights_list[0].keys()
        for key in target_keys:
            # Stack tensors for this key across the batch
            collated_targets[key] = torch.stack([t[key] for t in target_weights_list])

    # Return metadata as list of dicts, collated targets, and collated domain_ids
    return (
        metadata_list,  # List of dicts, not collated
        collated_targets,  # Dict of stacked tensors
        default_collate(domain_id_list),
    )


def curriculum_data_loader(
    dataset: Dataset,
    domain_to_id: Dict[str, int],
    batch_size: int = 4,
    current_stage: int = 1,
    num_stages: int = 5,
) -> DataLoader:
    """Create data loader that follows curriculum by domain difficulty.

    Stage 1: Only priority 1 domains (international_law, jurisprudence)
    Stage 2: Add priority 2 domains (medical)
    Stage 3: Add priority 3 domains (computer_science)
    Stage 4: Add priority 4 domains
    Stage 5: Add priority 5 domains (econometrics, statistics)
    """
    # Filter samples by current stage
    filtered_indices = []
    for idx in range(len(dataset)):
        metadata, _, domain_id = dataset[idx]
        metadata_domain = metadata.get("domain", "")
        # Find matching config by prefix (e.g., "medical" matches "medical_genetics")
        config = None
        for config_domain, config_obj in DOMAIN_CONFIGS.items():
            if metadata_domain.startswith(config_domain):
                config = config_obj
                break
        if config and config.priority <= current_stage:
            filtered_indices.append(idx)

    # Skip stages with 0 samples
    if len(filtered_indices) == 0:
        print(f"Warning: No samples for curriculum stage {current_stage}, skipping")
        return None

    filtered_dataset = torch.utils.data.Subset(dataset, filtered_indices)
    return DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=safe_metadata_collate,
    )


def train_hypernetwork(
    hypernetwork: DomainConditionedHypernetwork,
    encoder: StructuredMetadataEncoder,
    train_dataset: Dataset,
    val_loader: DataLoader,
    domain_to_id: Dict[str, int],
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    output_dir: str = "./checkpoints",
    use_curriculum: bool = True,
    num_curriculum_stages: int = 5,
    batch_size: int = 4,
    similarity_checker: Optional[EmbeddingSimilarityChecker] = None,
    cross_domain_evaluator: Optional[CrossDomainEvaluator] = None,
):
    """
    Train the hypernetwork to predict LoRA weights from metadata with advanced features.

    Args:
        hypernetwork: DomainConditionedHypernetwork model to train
        encoder: StructuredMetadataEncoder for encoding metadata
        train_dataset: Training dataset
        val_loader: Validation data loader
        domain_to_id: Mapping from domain names to IDs
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints
        use_curriculum: Whether to use curriculum learning by domain difficulty
        num_curriculum_stages: Number of curriculum stages
        batch_size: Batch size for data loaders
        similarity_checker: Optional embedding similarity checker
        cross_domain_evaluator: Optional cross-domain contamination evaluator
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move models to device
    hypernetwork = hypernetwork.to(device)
    encoder = encoder.to(device)

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        hypernetwork.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss function (MSE between predicted and target LoRA weights)
    criterion = nn.MSELoss()

    # Loglikelihood-aligned loss (weighted MSE)
    def loglikelihood_aligned_loss(
        pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss aligned with loglikelihood objective."""
        loss_a = criterion(pred["lora_A"], target["lora_A"])
        loss_b = criterion(pred["lora_B"], target["lora_B"])

        # Weight B matrix higher (affects output more directly)
        return loss_a + 2.0 * loss_b

    best_val_loss = float("inf")
    epochs_per_stage = (
        num_epochs // num_curriculum_stages if use_curriculum else num_epochs
    )

    # Create initial train loader
    if use_curriculum:
        train_loader = curriculum_data_loader(
            train_dataset,
            domain_to_id,
            batch_size,
            current_stage=1,
            num_stages=num_curriculum_stages,
        )
        # If stage 1 has no samples, try next stage
        if train_loader is None:
            for stage in range(2, num_curriculum_stages + 1):
                train_loader = curriculum_data_loader(
                    train_dataset,
                    domain_to_id,
                    batch_size,
                    current_stage=stage,
                    num_stages=num_curriculum_stages,
                )
                if train_loader is not None:
                    break
            if train_loader is None:
                raise ValueError("No samples available for any curriculum stage")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=safe_metadata_collate,
        )

    for epoch in range(num_epochs):
        # Determine curriculum stage
        if use_curriculum:
            current_stage = (epoch // epochs_per_stage) + 1
            current_stage = min(current_stage, num_curriculum_stages)

            # Recreate data loader for new stage
            new_train_loader = curriculum_data_loader(
                train_dataset,
                domain_to_id,
                batch_size,
                current_stage=current_stage,
                num_stages=num_curriculum_stages,
            )
            # Skip this epoch if stage has no samples
            if new_train_loader is None:
                print(
                    f"Skipping epoch {epoch + 1} - no samples for stage {current_stage}"
                )
                continue
            train_loader = new_train_loader
        else:
            current_stage = num_curriculum_stages

        # Training
        hypernetwork.train()
        encoder.eval()  # Keep encoder frozen
        train_loss = 0.0

        for batch_idx, (metadata_batch, target_batch, domain_id_batch) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()

            # Unpack target_batch from dict of stacked tensors to list of dicts
            target_weights_list = [
                {k: target_batch[k][i] for k in target_batch.keys()}
                for i in range(len(metadata_batch))
            ]

            batch_loss = 0.0
            for metadata, target_weights, domain_id in zip(
                metadata_batch, target_weights_list, domain_id_batch
            ):
                # Encode metadata
                metadata_emb = encoder(metadata)

                # Generate predicted LoRA weights
                pred_weights = hypernetwork(metadata_emb, domain_id)

                # Move target tensors to same device as predictions
                target_weights = {k: v.to(device) for k, v in target_weights.items()}

                # Compute loglikelihood-aligned loss
                loss = loglikelihood_aligned_loss(pred_weights, target_weights)
                batch_loss += loss

            batch_loss = batch_loss / len(metadata_batch)
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += batch_loss.item()

        train_loss /= len(train_loader)

        # Validation
        hypernetwork.eval()
        val_loss = 0.0

        with torch.no_grad():
            for metadata_batch, target_batch, domain_id_batch in val_loader:
                # Unpack target_batch from dict of stacked tensors to list of dicts
                target_weights_list = [
                    {k: target_batch[k][i] for k in target_batch.keys()}
                    for i in range(len(metadata_batch))
                ]

                batch_loss = 0.0
                for metadata, target_weights, domain_id in zip(
                    metadata_batch, target_weights_list, domain_id_batch
                ):
                    metadata_emb = encoder(metadata)
                    pred_weights = hypernetwork(metadata_emb, domain_id)

                    # Move target tensors to same device as predictions
                    target_weights = {
                        k: v.to(device) for k, v in target_weights.items()
                    }

                    loss = loglikelihood_aligned_loss(pred_weights, target_weights)
                    batch_loss += loss
                batch_loss = batch_loss / len(metadata_batch)
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)

        # Step scheduler
        scheduler.step()

        # Check embedding similarity
        similarity_report = {}
        if similarity_checker and epoch % 5 == 0:
            metadata_list = [m for m, _, _ in train_loader.dataset]
            similarity_report = similarity_checker.check_similarity(metadata_list)

        # Check cross-domain contamination
        contamination_report = {}
        if cross_domain_evaluator and epoch % 5 == 0:
            contamination_report = cross_domain_evaluator.evaluate_contamination(
                hypernetwork, encoder, device
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} (Stage {current_stage}/{num_curriculum_stages}) - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if similarity_report:
            print(
                f"  Embedding Similarity: {similarity_report['avg_similarity']:.3f} "
                f"(collapse: {similarity_report['collapse_detected']})"
            )

        if contamination_report:
            avg_contamination = np.mean(
                [
                    np.mean(list(scores.values()))
                    for scores in contamination_report.values()
                ]
            )
            print(f"  Cross-domain Contamination: {avg_contamination:.3f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, "best_hypernetwork.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "hypernetwork_state_dict": hypernetwork.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "current_stage": current_stage,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint to {checkpoint_path}")

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main():
    """Main training entry point with advanced asymmetric training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MetadataToLoRA hypernetwork with asymmetric training"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory containing metadata JSON files",
    )
    parser.add_argument(
        "--targets-dir",
        type=str,
        default=None,
        help="Directory containing target LoRA safetensors (if None, uses synthetic targets)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model identifier",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        default=False,
        help="Use curriculum learning by domain difficulty",
    )
    parser.add_argument(
        "--num-curriculum-stages",
        type=int,
        default=5,
        help="Number of curriculum stages",
    )
    parser.add_argument(
        "--check-similarity",
        action="store_true",
        default=True,
        help="Check embedding similarity for encoder collapse",
    )
    parser.add_argument(
        "--check-contamination",
        action="store_true",
        default=True,
        help="Check cross-domain contamination",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence encoder model",
    )

    args = parser.parse_args()

    # Get model dimensions
    model_dims = {
        "meta-llama/Llama-3-8B": (4096, 4096),
        "meta-llama/Llama-3-70B": (8192, 8192),
        "Qwen/Qwen2-7B": (3584, 3584),
        "deepseek-ai/DeepSeek-V3": (7168, 7168),
        "mistralai/Mistral-7B-Instruct-v0.2": (4096, 4096),
    }
    d_in, d_out = model_dims.get(args.base_model, (4096, 4096))

    # Create dataset
    if args.targets_dir:
        print("Using real target LoRA weights from fine-tuned models")
        dataset = LoRATargetDataset(args.metadata_dir, args.targets_dir)
    else:
        print(
            "Using synthetic target weights (domain-seeded with vocabulary distinctiveness)"
        )
        dataset = SyntheticTargetDataset(args.metadata_dir, args.base_model, args.rank)

    domain_to_id = dataset.domain_to_id
    num_domains = len(domain_to_id)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=safe_metadata_collate,
    )

    # Initialize sentence encoder
    print(f"Loading sentence encoder: {args.encoder_model}")
    base_encoder = SentenceTransformer(args.encoder_model)

    # Get encoder output dimension dynamically
    encoder_dim = base_encoder.get_sentence_embedding_dimension()
    print(f"Encoder output dimension: {encoder_dim}")

    # Initialize structured metadata encoder with dynamic dimension
    encoder = StructuredMetadataEncoder(base_encoder, embed_dim=encoder_dim)

    # Initialize domain-conditioned hypernetwork with dynamic dimension
    print(f"Initializing domain-conditioned hypernetwork with {num_domains} domains")
    hypernetwork = DomainConditionedHypernetwork(
        embed_dim=encoder_dim,
        rank=args.rank,
        d_in=d_in,
        d_out=d_out,
        hidden_dim=2048,
        num_domains=num_domains,
    )

    # Initialize optional monitoring tools
    similarity_checker = None
    if args.check_similarity:
        print("Enabling embedding similarity checker")
        similarity_checker = EmbeddingSimilarityChecker(base_encoder)

    cross_domain_evaluator = None
    if args.check_contamination:
        print("Enabling cross-domain contamination evaluator")
        cross_domain_evaluator = CrossDomainEvaluator(args.metadata_dir)

    # Train
    print(
        f"\nStarting training with {len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )
    print(f"Domains: {list(domain_to_id.keys())}")
    print(
        f"Domain configs: {[(d, DOMAIN_CONFIGS.get(d, None).difficulty if DOMAIN_CONFIGS.get(d) else 'N/A') for d in domain_to_id.keys()]}"
    )
    print()

    train_hypernetwork(
        hypernetwork,
        encoder,
        train_dataset,
        val_loader,
        domain_to_id,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        use_curriculum=args.use_curriculum,
        num_curriculum_stages=args.num_curriculum_stages,
        batch_size=args.batch_size,
        similarity_checker=similarity_checker,
        cross_domain_evaluator=cross_domain_evaluator,
    )


if __name__ == "__main__":
    main()
