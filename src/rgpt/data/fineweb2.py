"""
src/rgpt/data/fineweb2.py : Streaming ingestion and cleaning of FineWeb-2 Bengali subset (ben_Beng).
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §1
Contract      : target_bytes: int -> Iterator[dict]
Invariants    : Streams from HuggingFace; E-gen drawn deterministically from test split BEFORE dedup
Failure modes : HuggingFace Hub network failure; dataset schema mismatch
Status        : TESTED
"""

import io
import json
import os
import urllib.request
from typing import Any, Dict, Iterable, Iterator, Optional

from src.rgpt.text.clean import clean_pipeline


def _stream_hf_parquet(
    dataset_name: str, subset: str, split: str
) -> Iterator[Dict[str, Any]]:
    """Stream parquet files directly from HuggingFace dataset repo using pyarrow."""
    import pyarrow.parquet as pq

    contact = os.environ.get("RGPT_CONTACT") or "asmitsarkar@users.noreply.github.com"
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet/{subset}"
    req = urllib.request.Request(
        api_url, headers={"User-Agent": f"RabindricGPT/1.0 ({contact})"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch parquet list for {dataset_name}/{subset}: {e}"
        ) from e

    file_urls = data.get(split, [])
    for file_url in file_urls:
        file_req = urllib.request.Request(
            file_url, headers={"User-Agent": f"RabindricGPT/1.0 ({contact})"}
        )
        with urllib.request.urlopen(file_req) as file_resp:
            pf = pq.ParquetFile(io.BytesIO(file_resp.read()))
            for batch in pf.iter_batches(batch_size=256):
                col_names = batch.schema.names
                texts = batch["text"].to_pylist()
                ids = (
                    batch["id"].to_pylist()
                    if "id" in col_names
                    else [str(i) for i in range(len(texts))]
                )
                for doc_id, text in zip(ids, texts):
                    yield {"id": doc_id, "text": text}


def stream_fineweb2_bengali(
    dataset_name: str = "HuggingFaceFW/fineweb-2",
    subset: str = "ben_Beng",
    split: str = "train",
    max_bytes: float = 5.0e9,
    source_iterable: Optional[Iterable[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream and yield cleaned documents from FineWeb-2 Bengali up to max_bytes.

    Args:
        dataset_name: HuggingFace repository ID.
        subset: Language/script identifier.
        split: Dataset split ('train' or 'test').
        max_bytes: Target volume in cleaned bytes.
        source_iterable: Optional generator/iterable for offline/mock execution in unit tests.

    Yields:
        Document dictionaries containing cleaned text and metadata.
    """
    if source_iterable is None:
        try:
            from datasets import load_dataset

            stream = load_dataset(dataset_name, subset, split=split, streaming=True)
        except (ImportError, ModuleNotFoundError):
            stream = _stream_hf_parquet(dataset_name, subset, split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to stream {dataset_name}/{subset} split {split}: {e}"
            ) from e
    else:
        stream = source_iterable

    accumulated_bytes = 0
    for doc in stream:
        raw_text = doc.get("text", "")
        if not raw_text:
            continue

        clean_text, passes, metrics = clean_pipeline(raw_text, threshold=0.80)
        if not passes or not clean_text:
            continue

        doc_bytes = len(clean_text.encode("utf-8"))
        accumulated_bytes += doc_bytes

        yield {
            "id": doc.get("id", str(accumulated_bytes)),
            "text": clean_text,
            "bytes": doc_bytes,
            "chars": len(clean_text),
            "bengali_ratio": metrics.get("bengali_ratio", 1.0),
        }

        if accumulated_bytes >= max_bytes:
            break


def sample_egen_docs(
    dataset_name: str = "HuggingFaceFW/fineweb-2",
    subset: str = "ben_Beng",
    target_chars: int = 2000000,
    seed: int = 42,
    source_iterable: Optional[Iterable[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Sample held-out E-gen intrinsic evaluation documents from official test split.

    Args:
        dataset_name: HuggingFace repository ID.
        subset: Language/script identifier.
        target_chars: Target character count for E-gen (~2,000,000 characters).
        seed: Random seed for deterministic document order.
        source_iterable: Optional generator/iterable for offline execution.

    Yields:
        Cleaned held-out evaluation documents.
    """
    accumulated_chars = 0
    for doc in stream_fineweb2_bengali(
        dataset_name=dataset_name,
        subset=subset,
        split="test",
        max_bytes=target_chars * 4,  # Upper bound on bytes
        source_iterable=source_iterable,
    ):
        yield doc
        accumulated_chars += doc["chars"]
        if accumulated_chars >= target_chars:
            break
