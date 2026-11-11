
import os
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset, load_from_disk
except Exception:
    load_dataset = None
    load_from_disk = None


class FlexibleDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        model=None,
        seq_len=256,
        num_samples=128,
        model_name=None,
    ):
        self.dataset_arg = dataset
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)
        self.model_name = model_name
        self.model = model
        self.is_encoder_decoder = bool(
            getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        )
        self.dataset = None
        self.use_synthetic = False

        print(
            f"[FlexibleDataset] model_type="
            f"{getattr(getattr(model, 'config', None), 'model_type', None)}, "
            f"is_encoder_decoder={self.is_encoder_decoder}",
            flush=True,
        )

        dataset_path = str(dataset)

        # Case 1: dataset is a local path.
        if os.path.exists(dataset_path):
            print(f"[FlexibleDataset] loading local dataset from {dataset_path}", flush=True)

            # Try HuggingFace load_from_disk first.
            if load_from_disk is not None:
                try:
                    ds = load_from_disk(dataset_path)
                    if isinstance(ds, dict):
                        if "train" in ds:
                            ds = ds["train"]
                        else:
                            ds = list(ds.values())[0]
                    self.dataset = ds
                    print("[FlexibleDataset] loaded local dataset by load_from_disk", flush=True)
                except Exception as e:
                    print(f"[FlexibleDataset] load_from_disk failed: {type(e).__name__}: {e}", flush=True)

            # If it is a HuggingFace cache directory, try load_dataset wikitext from cache.
            if self.dataset is None and load_dataset is not None:
                try:
                    self.dataset = load_dataset(
                        "wikitext",
                        "wikitext-2-raw-v1",
                        split=f"train[:{self.num_samples}]",
                        cache_dir=dataset_path,
                        download_mode="reuse_cache_if_exists",
                    )
                    print("[FlexibleDataset] loaded wikitext from local HF cache", flush=True)
                except Exception as e:
                    print(f"[FlexibleDataset] load_dataset wikitext cache failed: {type(e).__name__}: {e}", flush=True)

            # Try local text files.
            if self.dataset is None and load_dataset is not None:
                text_files = []
                for suffix in ["*.txt", "*.text", "*.json", "*.jsonl"]:
                    text_files.extend(str(p) for p in Path(dataset_path).rglob(suffix))

                if text_files:
                    try:
                        if any(p.endswith((".json", ".jsonl")) for p in text_files):
                            self.dataset = load_dataset(
                                "json",
                                data_files=text_files,
                                split="train",
                            )
                        else:
                            self.dataset = load_dataset(
                                "text",
                                data_files=text_files,
                                split="train",
                            )
                        print(f"[FlexibleDataset] loaded local text/json files, num_files={len(text_files)}", flush=True)
                    except Exception as e:
                        print(f"[FlexibleDataset] local text/json loading failed: {type(e).__name__}: {e}", flush=True)

        # Case 2: dataset is not a path. Try the requested dataset name.
        if self.dataset is None and load_dataset is not None:
            try:
                cache_dir = os.environ.get(
                    "HF_DATASETS_CACHE",
                    os.path.expanduser("~/.cache/harmoeny/datasets"),
                )

                if str(dataset).lower() == "wikitext":
                    self.dataset = load_dataset(
                        "wikitext",
                        "wikitext-2-raw-v1",
                        split=f"train[:{self.num_samples}]",
                        cache_dir=cache_dir,
                        download_mode="reuse_cache_if_exists",
                    )
                    print("[FlexibleDataset] loaded wikitext dataset", flush=True)
                else:
                    self.dataset = load_dataset(
                        "glue",
                        "sst2",
                        split=f"train[:{self.num_samples}]",
                        streaming=False,
                        cache_dir=cache_dir,
                        download_mode="reuse_cache_if_exists",
                    )
                    print("[FlexibleDataset] loaded glue/sst2 dataset", flush=True)
            except Exception as e:
                print(f"[FlexibleDataset] HF dataset loading failed: {type(e).__name__}: {e}", flush=True)

        if self.dataset is None:
            print("[FlexibleDataset] fallback to synthetic dataset", flush=True)
            self.use_synthetic = True

        self.synthetic_text = (
            "This is a synthetic sentence for offline QoSMoE inference benchmarking. "
            "It is repeated to create a stable input sequence. "
        )

    def __len__(self):
        if self.use_synthetic or self.dataset is None:
            return self.num_samples
        try:
            return min(len(self.dataset), self.num_samples)
        except Exception:
            return self.num_samples

    def _encode_text(self, text):
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}

        # Only encoder-decoder models (e.g. Switch Transformer) need
        # decoder_input_ids / decoder_attention_mask. Causal LMs such as
        # Qwen1.5-MoE must NOT receive these kwargs.
        if self.is_encoder_decoder:
            if "input_ids" in item and "decoder_input_ids" not in item:
                item["decoder_input_ids"] = item["input_ids"].clone()

            if "attention_mask" in item and "decoder_attention_mask" not in item:
                item["decoder_attention_mask"] = item["attention_mask"].clone()

        return item

    def __getitem__(self, idx):
        if self.use_synthetic or self.dataset is None:
            text = (self.synthetic_text * ((self.seq_len // 16) + 2))[: max(32, self.seq_len * 8)]
            return self._encode_text(text)

        item = self.dataset[int(idx)]

        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("sentence")
                or item.get("content")
                or item.get("article")
                or str(item)
            )
        else:
            text = str(item)

        if text is None or len(str(text).strip()) == 0:
            text = self.synthetic_text

        return self._encode_text(str(text))
