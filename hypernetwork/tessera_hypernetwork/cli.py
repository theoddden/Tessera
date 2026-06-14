"""
Tessera Hypernetwork CLI
Command-line interface for LoRA adapter generation and serving
"""

import click
import json
from pathlib import Path


@click.group()
def cli():
    """Tessera Hypernetwork CLI - Generate and serve LoRA adapters"""
    pass


@cli.command()
@click.option('--from-metadata', type=str, help='JSON metadata string or file path')
@click.option('--from-text', type=str, help='Natural language description')
@click.option('--from-doc', type=str, help='Document content or file path')
@click.option('--base-model', type=str, required=True, help='Base model identifier (e.g., meta-llama/Llama-3-8B)')
@click.option('--rank', type=int, default=16, help='LoRA rank (default: 16)')
@click.option('--save', type=str, required=True, help='Output path for safetensors file')
@click.option('--mode', type=str, default=None, help='Generation mode: doc, metadata, or text (auto-inferred if not specified)')
def generate(from_metadata, from_text, from_doc, base_model, rank, save, mode):
    """Generate LoRA adapter from metadata, text, or document"""
    
    # Validate input
    inputs_provided = sum([bool(from_metadata), bool(from_text), bool(from_doc)])
    if inputs_provided == 0:
        click.echo("Error: Must provide one of --from-metadata, --from-text, or --from-doc", err=True)
        raise click.Abort()
    if inputs_provided > 1:
        click.echo("Error: Must provide exactly one of --from-metadata, --from-text, or --from-doc", err=True)
        raise click.Abort()
    
    # Determine mode
    if mode is None:
        if from_metadata:
            mode = 'metadata'
        elif from_doc:
            mode = 'doc'
        else:
            mode = 'text'
    
    # Load input content
    content = None
    if from_metadata:
        # Try parsing as JSON first, then check if it's a file path
        try:
            content = json.loads(from_metadata)
        except json.JSONDecodeError:
            # If JSON parsing fails, try as file path
            try:
                with open(from_metadata, 'r') as f:
                    content = json.load(f)
            except (FileNotFoundError, IOError, json.JSONDecodeError):
                click.echo("Error: --from-metadata must be valid JSON or a file path containing JSON", err=True)
                raise click.Abort()
    elif from_doc:
        # Try as file path first, then use as direct content
        try:
            with open(from_doc, 'r') as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            content = from_doc
    else:  # from_text
        content = from_text
    
    # Generate LoRA weights
    click.echo(f"Generating LoRA adapter (mode={mode}, rank={rank}, base_model={base_model})...")
    
    # Lazy-load heavy dependencies
    from safetensors.torch import save_file
    from tessera_hypernetwork.doc_to_lora import DocToLoRA
    from tessera_hypernetwork.metadata_to_lora import MetadataToLoRA
    from tessera_hypernetwork.text_to_lora import TextToLoRA
    
    try:
        if mode == 'metadata':
            generator = MetadataToLoRA(base_model, default_rank=rank)
            lora_weights = generator.generate(content, rank)
        elif mode == 'doc':
            generator = DocToLoRA(base_model, use_shine=True, default_rank=rank)
            lora_weights = generator.generate(content, rank)
        else:  # text
            generator = TextToLoRA(base_model, default_rank=rank)
            lora_weights = generator.generate(content, rank)
        
        # Save to safetensors
        output_path = Path(save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_weights, str(output_path))
        
        click.echo(f"✓ LoRA adapter saved to {output_path}")
        click.echo(f"  - lora_A shape: {lora_weights['lora_A'].shape}")
        click.echo(f"  - lora_B shape: {lora_weights['lora_B'].shape}")
        
    except Exception as e:
        click.echo(f"Error generating LoRA adapter: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--port', type=int, default=8000, help='Port to serve on (default: 8000)')
@click.option('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
@click.option('--qdrant-url', type=str, default=None, help='Qdrant vector database URL (optional)')
@click.option('--workers', type=int, default=1, help='Number of worker processes (default: 1)')
def serve(port, host, qdrant_url, workers):
    """Start the Tessera hypernetwork server"""
    
    # Lazy-load uvicorn and server
    import uvicorn
    from tessera_hypernetwork.server import app
    
    if qdrant_url:
        click.echo(f"Starting Tessera server on {host}:{port} with Qdrant at {qdrant_url}")
    else:
        click.echo(f"Starting Tessera server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.command()
@click.option('--url', type=str, default='http://localhost:8000', help='Server URL (default: http://localhost:8000)')
def health(url):
    """Check server health status"""
    
    import requests
    
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo(f"✓ Server healthy: {data}")
        else:
            click.echo(f"✗ Server returned status {response.status_code}", err=True)
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to server: {e}", err=True)
        raise click.Abort()


@cli.command()
def list():
    """List available base models and their dimensions"""

    model_dims = {
        "meta-llama/Llama-3-8B": (4096, 4096),
        "meta-llama/Llama-3-70B": (8192, 8192),
        "Qwen/Qwen2-7B": (3584, 3584),
        "deepseek-ai/DeepSeek-V3": (7168, 7168),
    }

    click.echo("Available base models:")
    click.echo("")
    for model, (d_in, d_out) in model_dims.items():
        click.echo(f"  {model}")
        click.echo(f"    Dimensions: {d_in} x {d_out}")
    click.echo("")
    click.echo("Default dimensions for unknown models: 4096 x 4096")


# LoRAX commands
@cli.group()
def lorax():
    """LoRAX adapter management commands"""
    pass


@lorax.command()
@click.option('--path', type=str, required=True, help='Path to the adapter safetensors file')
@click.option('--name', type=str, required=True, help='Name for the adapter in LoRAX')
@click.option('--base-model', type=str, required=True, help='Base model identifier')
@click.option('--lorax-url', type=str, default='http://localhost:8080', help='LoRAX server URL (default: http://localhost:8080)')
def import_adapter(path, name, base_model, lorax_url):
    """Import an adapter into LoRAX"""

    import requests

    click.echo(f"Importing adapter '{name}' from {path} to LoRAX at {lorax_url}")

    # Read the adapter file
    try:
        with open(path, 'rb') as f:
            adapter_data = f.read()
    except FileNotFoundError:
        click.echo(f"Error: Adapter file not found at {path}", err=True)
        raise click.Abort()

    # Send to LoRAX
    try:
        response = requests.post(
            f"{lorax_url}/v1/adapters",
            files={"file": (Path(path).name, adapter_data, "application/octet-stream")},
            data={
                "adapter_name": name,
                "base_model": base_model,
            },
            timeout=30
        )

        if response.status_code == 200:
            click.echo(f"✓ Adapter '{name}' imported successfully")
        else:
            click.echo(f"✗ Failed to import adapter: {response.status_code} - {response.text}", err=True)
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to LoRAX: {e}", err=True)
        raise click.Abort()


@lorax.command()
@click.option('--lorax-url', type=str, default='http://localhost:8080', help='LoRAX server URL (default: http://localhost:8080)')
def list_adapters(lorax_url):
    """List adapters loaded in LoRAX"""

    import requests

    try:
        response = requests.get(f"{lorax_url}/v1/adapters", timeout=10)

        if response.status_code == 200:
            adapters = response.json()
            if adapters:
                click.echo("Loaded adapters:")
                for adapter in adapters:
                    click.echo(f"  - {adapter.get('name', 'unknown')}: {adapter.get('base_model', 'unknown')}")
            else:
                click.echo("No adapters loaded")
        else:
            click.echo(f"✗ Failed to list adapters: {response.status_code} - {response.text}", err=True)
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to LoRAX: {e}", err=True)
        raise click.Abort()


@lorax.command()
@click.option('--name', type=str, required=True, help='Name of the adapter to unload')
@click.option('--lorax-url', type=str, default='http://localhost:8080', help='LoRAX server URL (default: http://localhost:8080)')
def unload(name, lorax_url):
    """Unload an adapter from LoRAX"""

    import requests

    click.echo(f"Unloading adapter '{name}' from LoRAX at {lorax_url}")

    try:
        response = requests.delete(f"{lorax_url}/v1/adapters/{name}", timeout=10)

        if response.status_code == 200:
            click.echo(f"✓ Adapter '{name}' unloaded successfully")
        else:
            click.echo(f"✗ Failed to unload adapter: {response.status_code} - {response.text}", err=True)
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to LoRAX: {e}", err=True)
        raise click.Abort()


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()
