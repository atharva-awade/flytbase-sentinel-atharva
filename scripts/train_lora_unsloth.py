"""LoRA fine-tune Qwen3-VL-4B-Instruct on the SFT set with Unsloth (PLAN.md §3.3). Kaggle T4 / Modal L4.

  pip install "unsloth[colab-new]" trl peft accelerate bitsandbytes  (T4: fp16, no bf16)
  python scripts/train_lora_unsloth.py --data data/sft --out output/qwen3vl4b-sentinel-lora --epochs 1

Exports: LoRA adapter (for vLLM --enable-lora), merged fp16 (optional), GGUF Q4_K_M (optional, edge demo).
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from PIL import Image

from sentinel.prompts import SYSTEM

app = typer.Typer(add_completion=False)


def load_rows(p: Path, max_pixels: int):
    rows = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        content = [{"type": "image", "image": Image.open(x).convert("RGB")} for x in r["images"]]
        content.append({"type": "text", "text": r["question"]})
        rows.append({"messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
        ]})
    return rows


@app.command()
def main(data: Path = "data/sft", out: Path = "output/qwen3vl4b-sentinel-lora",
         base: str = "unsloth/Qwen3-VL-4B-Instruct", epochs: float = 1.0, lr: float = 1e-4, r: int = 16,
         batch: int = 1, grad_accum: int = 4, max_pixels: int = 384 * 28 * 28, min_pixels: int = 128 * 28 * 28,
         load_4bit: bool = True, export_gguf: bool = False, merge_fp16: bool = False, max_steps: int = -1):
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator

    model, tok = FastVisionModel.from_pretrained(base, load_in_4bit=load_4bit, use_gradient_checkpointing="unsloth")
    # image token budget = the VRAM/latency dial
    tok.image_processor.min_pixels = min_pixels
    tok.image_processor.max_pixels = max_pixels
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=r, lora_alpha=r, lora_dropout=0, bias="none", random_state=0,
    )
    train = load_rows(data / "train.jsonl", max_pixels)          # list comprehension, NOT dataset.map
    val = load_rows(data / "val.jsonl", max_pixels) if (data / "val.jsonl").exists() else None
    FastVisionModel.for_training(model)
    bf16 = is_bf16_supported()
    trainer = SFTTrainer(
        model=model, tokenizer=tok, data_collator=UnslothVisionDataCollator(model, tok),
        train_dataset=train, eval_dataset=val,
        args=SFTConfig(
            per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.03, num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr,
            fp16=not bf16, bf16=bf16, logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=0, output_dir=str(out / "ckpt"), report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,                       # else image tokens get truncated silently
            save_steps=100, save_total_limit=2,
            eval_strategy="steps" if val else "no", eval_steps=100,
        ),
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}  bf16={bf16}  train={len(train)} val={len(val) if val else 0}")
    trainer.train()
    model.save_pretrained(str(out)); tok.save_pretrained(str(out))
    print("adapter →", out)
    if merge_fp16:
        model.save_pretrained_merged(str(out) + "-merged", tok, save_method="merged_16bit")
    if export_gguf:
        model.save_pretrained_gguf(str(out) + "-gguf", tok, quantization_method="q4_k_m")


if __name__ == "__main__":
    app()
