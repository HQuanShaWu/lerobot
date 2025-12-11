from pathlib import Path

from huggingface_hub import hf_hub_download


def ensure_robobrain_cache_ready(cache_dir: Path, assets_repo: str) -> None:
    """Ensure RoboBrain processor/tokenizer assets exist in cache_dir."""
    cache_dir = Path(cache_dir)

    required_assets = [
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]

    print(f"[EFFICIENTVLA] Assets repo: {assets_repo} \n Cache dir: {cache_dir}")

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            print(f"[EFFICIENTVLA] Fetching {fname}")
            hf_hub_download(
                repo_id=assets_repo,
                filename=fname,
                repo_type="model",
                local_dir=str(cache_dir),
            )
