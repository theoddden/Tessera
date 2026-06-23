#!/usr/bin/env python3
"""
Tessera v2.1.0 Training Data Preparation
Downloads domain-specific datasets from HuggingFace and prepares them as training targets
"""

import os
import json
import random
from pathlib import Path

# Install required packages first:
# pip install datasets transformers peft torch accelerate

try:
    from datasets import load_dataset
    print("✓ datasets imported")
except ImportError:
    print("Run: pip install datasets")
    exit(1)

OUTPUT_DIR = Path("./training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

DOMAINS = {
    "legal": {
        "datasets": [
            {
                "name": "pile-of-law/pile-of-law",
                "config": "courtlistener_opinions",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            },
        ],
        "fallback": [
            {
                "name": "nguyen-brat/legal_contracts",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            }
        ]
    },
    "medical": {
        "datasets": [
            {
                "name": "pubmed_qa",
                "config": "pqa_labeled",
                "split": "train",
                "text_field": "long_answer",
                "samples": 500,
            },
            {
                "name": "medalpaca/medical_meadow_medqa",
                "split": "train",
                "text_field": "output",
                "samples": 250,
            },
        ],
        "fallback": []
    },
    "code": {
        "datasets": [
            {
                "name": "codeparrot/github-code",
                "config": "Python",
                "split": "train",
                "text_field": "code",
                "samples": 500,
                "streaming": True,
            },
        ],
        "fallback": [
            {
                "name": "flytech/python-codes-25k",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            }
        ]
    },
    "finance": {
        "datasets": [
            {
                "name": "financial_phrasebank",
                "config": "sentences_allagree",
                "split": "train",
                "text_field": "sentence",
                "samples": 500,
            },
            {
                "name": "TheFinAI/flare-finqa",
                "split": "train",
                "text_field": "query",
                "samples": 250,
            },
        ],
        "fallback": []
    },
    "security": {
        "datasets": [
            {
                "name": "mrm8488/llm-attacks-dataset",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            },
        ],
        "fallback": [
            {
                "name": "CyberNative-AI/Cybersecurity-Data",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            }
        ]
    },
    "academic": {
        "datasets": [
            {
                "name": "allenai/peS2o",
                "config": "v2",
                "split": "train",
                "text_field": "text",
                "samples": 500,
                "streaming": True,
            },
        ],
        "fallback": [
            {
                "name": "scientific_papers",
                "config": "arxiv",
                "split": "train",
                "text_field": "abstract",
                "samples": 500,
            }
        ]
    },
    "ml": {
        "datasets": [
            {
                "name": "ought/raft",
                "config": "terms_of_service",
                "split": "train",
                "text_field": "Tweet",
                "samples": 500,
            },
        ],
        "fallback": [
            {
                "name": "teknium/OpenHermes-2.5",
                "split": "train",
                "text_field": "conversations",
                "samples": 500,
            }
        ]
    },
    "biotech": {
        "datasets": [
            {
                "name": "bigbio/pubmed_qa",
                "config": "pubmed_qa_labeled_fold0_bigbio_qa",
                "split": "train",
                "text_field": "answers",
                "samples": 500,
            },
        ],
        "fallback": [
            {
                "name": "AI-BIO/Medical-NLP",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            }
        ]
    },
    "devops": {
        "datasets": [
            {
                "name": "smangrul/hf-stack-v1",
                "split": "train",
                "text_field": "content",
                "samples": 500,
                "streaming": True,
            },
        ],
        "fallback": [
            {
                "name": "flytech/python-codes-25k",
                "split": "train",
                "text_field": "text",
                "samples": 500,
            }
        ]
    },
    "hardware": {
        "datasets": [
            {
                "name": "nvidia/OpenMathInstruct-2",
                "split": "train",
                "text_field": "problem",
                "samples": 500,
                "streaming": True,
            },
        ],
        "fallback": [
            {
                "name": "math_dataset",
                "config": "algebra__linear_1d",
                "split": "train",
                "text_field": "question",
                "samples": 500,
            }
        ]
    },
}


def load_samples(dataset_config: dict, n_samples: int) -> list[str]:
    """Load text samples from a HuggingFace dataset."""
    name = dataset_config["name"]
    config = dataset_config.get("config")
    split = dataset_config.get("split", "train")
    text_field = dataset_config["text_field"]
    streaming = dataset_config.get("streaming", False)
    samples = []

    try:
        print(f"  Loading {name} ({config or 'default'})...")
        if streaming:
            ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
            for i, row in enumerate(ds):
                if i >= n_samples * 2:
                    break
                text = row.get(text_field, "")
                if isinstance(text, list):
                    text = " ".join([str(t) for t in text])
                if isinstance(text, dict):
                    text = json.dumps(text)
                text = str(text).strip()
                if len(text) > 50:
                    samples.append(text)
                if len(samples) >= n_samples:
                    break
        else:
            ds = load_dataset(name, config, split=split, trust_remote_code=True)
            indices = random.sample(range(len(ds)), min(n_samples * 2, len(ds)))
            for idx in indices:
                row = ds[idx]
                text = row.get(text_field, "")
                if isinstance(text, list):
                    text = " ".join([str(t) for t in text])
                if isinstance(text, dict):
                    text = json.dumps(text)
                text = str(text).strip()
                if len(text) > 50:
                    samples.append(text)
                if len(samples) >= n_samples:
                    break

        print(f"  ✓ Loaded {len(samples)} samples from {name}")
        return samples

    except Exception as e:
        print(f"  ✗ Failed to load {name}: {e}")
        return []


def prepare_domain(domain: str, config: dict) -> int:
    """Download and save samples for a domain."""
    domain_dir = OUTPUT_DIR / domain
    domain_dir.mkdir(exist_ok=True)
    output_file = domain_dir / "corpus.jsonl"

    if output_file.exists():
        existing = sum(1 for _ in open(output_file))
        if existing >= 400:
            print(f"  ✓ {domain}: already has {existing} samples, skipping")
            return existing

    all_samples = []
    datasets_to_try = config["datasets"] + config.get("fallback", [])

    for dataset_config in datasets_to_try:
        n_needed = 500 - len(all_samples)
        if n_needed <= 0:
            break
        samples = load_samples(dataset_config, n_needed)
        all_samples.extend(samples)

    if not all_samples:
        print(f"  ✗ {domain}: no samples loaded from any source")
        return 0

    # Shuffle and deduplicate
    random.shuffle(all_samples)
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = s[:100]
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)

    # Save as JSONL
    with open(output_file, "w") as f:
        for i, text in enumerate(unique_samples[:500]):
            f.write(json.dumps({
                "id": f"{domain}_{i:04d}",
                "domain": domain,
                "text": text,
                "length": len(text),
            }) + "\n")

    print(f"  ✓ {domain}: saved {len(unique_samples[:500])} samples to {output_file}")
    return len(unique_samples[:500])


def create_metadata_packets(domain: str, n_packets: int = 5) -> list[dict]:
    """Create rich metadata packets for each domain."""
    domain_profiles = {
        "legal": [
            {"role": "Senior Associate", "specialty": "Commercial litigation", "tools": ["Westlaw", "PACER"]},
            {"role": "Patent Attorney", "specialty": "Software patents", "tools": ["Docketbird", "Patent Center"]},
            {"role": "In-House Counsel", "specialty": "SaaS contracts", "tools": ["Ironclad", "DocuSign"]},
            {"role": "Criminal Defense Attorney", "specialty": "Federal criminal defense", "tools": ["Westlaw", "PACER"]},
            {"role": "Regulatory Counsel", "specialty": "FDA compliance", "tools": ["Westlaw", "FDA databases"]},
        ],
        "medical": [
            {"role": "Clinical Researcher", "specialty": "Oncology trials", "tools": ["REDCap", "SAS"]},
            {"role": "Genomics Scientist", "specialty": "Variant analysis", "tools": ["GATK", "IGV"]},
            {"role": "Epidemiologist", "specialty": "Disease surveillance", "tools": ["R", "ArcGIS"]},
            {"role": "Pathologist", "specialty": "Digital pathology", "tools": ["QuPath", "Aperio"]},
            {"role": "Clinical Trial Manager", "specialty": "Phase III trials", "tools": ["Medidata", "CTMS"]},
        ],
        "code": [
            {"role": "Backend Engineer", "specialty": "Python microservices", "tools": ["FastAPI", "PostgreSQL"]},
            {"role": "Frontend Engineer", "specialty": "React TypeScript", "tools": ["React", "TypeScript"]},
            {"role": "ML Engineer", "specialty": "LLM inference", "tools": ["vLLM", "PyTorch"]},
            {"role": "Security Engineer", "specialty": "Application security", "tools": ["Burp Suite", "Semgrep"]},
            {"role": "Infrastructure Engineer", "specialty": "Kubernetes", "tools": ["Terraform", "Helm"]},
        ],
        "finance": [
            {"role": "Quantitative Analyst", "specialty": "Derivatives pricing", "tools": ["Python", "Bloomberg"]},
            {"role": "Equity Researcher", "specialty": "Tech sector", "tools": ["FactSet", "Excel"]},
            {"role": "Credit Analyst", "specialty": "Corporate credit", "tools": ["Moody's", "S&P Capital IQ"]},
            {"role": "Private Equity Associate", "specialty": "LBO modeling", "tools": ["Excel", "CapIQ"]},
            {"role": "Macro Strategist", "specialty": "FX and rates", "tools": ["Bloomberg", "Refinitiv"]},
        ],
        "security": [
            {"role": "Penetration Tester", "specialty": "Web application testing", "tools": ["Burp Suite", "Metasploit"]},
            {"role": "SOC Analyst", "specialty": "Threat detection", "tools": ["Splunk", "CrowdStrike"]},
            {"role": "Cloud Security Engineer", "specialty": "AWS security", "tools": ["AWS Security Hub", "Prowler"]},
            {"role": "GRC Analyst", "specialty": "SOC2 compliance", "tools": ["Drata", "Vanta"]},
            {"role": "AI Security Researcher", "specialty": "LLM red teaming", "tools": ["Garak", "custom scripts"]},
        ],
    }

    profiles = domain_profiles.get(domain, [
        {"role": f"{domain.title()} Specialist", "specialty": domain, "tools": []}
        for _ in range(n_packets)
    ])

    packets = []
    for i, profile in enumerate(profiles[:n_packets]):
        packets.append({
            "id": f"{domain}_{i+1:03d}",
            "domain": f"{domain}_{profile['specialty'].lower().replace(' ', '_')[:20]}",
            "role": profile["role"],
            "specialty": profile["specialty"],
            "tools": profile["tools"],
            "current_project": f"Active {domain} project requiring domain expertise",
        })
    return packets


def main():
    print("=" * 60)
    print("TESSERA v2.1.0 TRAINING DATA PREPARATION")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    random.seed(42)
    total_samples = 0
    all_metadata = []

    for domain, config in DOMAINS.items():
        print(f"\n{'─'*40}")
        print(f"Domain: {domain.upper()}")
        print(f"{'─'*40}")
        n = prepare_domain(domain, config)
        total_samples += n

        # Generate enhanced metadata packets
        packets = create_metadata_packets(domain, n_packets=5)
        all_metadata.extend(packets)

    # Save combined metadata
    metadata_file = OUTPUT_DIR / "metadata_packets_v2.py"
    with open(metadata_file, "w") as f:
        f.write("METADATA_PACKETS = ")
        f.write(json.dumps(all_metadata, indent=2))
    print(f"\n✓ Saved {len(all_metadata)} metadata packets to {metadata_file}")

    # Save summary
    summary = {
        "total_samples": total_samples,
        "total_metadata_packets": len(all_metadata),
        "domains": list(DOMAINS.keys()),
        "per_domain": {
            domain: sum(1 for _ in open(OUTPUT_DIR / domain / "corpus.jsonl"))
            for domain in DOMAINS.keys()
            if (OUTPUT_DIR / domain / "corpus.jsonl").exists()
        }
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total corpus samples:    {total_samples}")
    print(f"Total metadata packets:  {len(all_metadata)}")
    print(f"Domains prepared:        {len(DOMAINS)}")
    print(f"\nNext step: run hypernetwork training with:")
    print(f"python3 -m tessera_hypernetwork.train_hypernetwork \\")
    print(f"  --metadata-dir {OUTPUT_DIR}/metadata_packets_v2.py \\")
    print(f"  --corpus-dir {OUTPUT_DIR} \\")
    print(f"  --rank 16 --epochs 100 \\")
    print(f"  --output-dir ./checkpoints_v2")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()