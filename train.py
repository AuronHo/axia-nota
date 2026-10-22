"""Fine-tune Qwen2.5-1.5B-Instruct into Bu Melati with LoRA, on your own NVIDIA GPU.

  python train.py                      # defaults: 3 epochs, LoRA r=16, saves out/lora and out/merged
  python train.py --epochs 2 --batch 2 # less VRAM
  python train.py --base Qwen/Qwen2.5-0.5B-Instruct   # smaller/faster variant

VRAM: defaults (batch 2 x accum 8) fit an 8 GB card like the RTX 5060; use --batch 1 --grad-accum 16 if you run out.
Only the assistant's answer is trained on; the game prompt is context.
"""
import argparse, json, os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default=str(HERE / "data/train.jsonl"))
    ap.add_argument("--val", default=str(HERE / "data/val.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Install the CUDA build of PyTorch (see SETUP.md) or train on Colab/Kaggle.")
    gpu = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {gpu}  | precision: {'bf16' if bf16 else 'fp16'}")

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or "<|endoftext|>"
    tok.padding_side = "right"

    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = load_dataset("json", data_files={"train": a.train, "validation": a.val})
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    lens = [len(tok(t).input_ids) for t in ds["train"]["text"][:200]]
    print(f"train={len(ds['train'])} val={len(ds['validation'])}  | tokens per example ~{sum(lens)//max(1,len(lens))} (max {max(lens)})")

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False

    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Train only on what Bu Melati says (everything after "<|im_start|>assistant\n").
    collator = DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant\n", tokenizer=tok)

    cfg = SFTConfig(
        output_dir=str(Path(a.out) / "checkpoints"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16,
        max_seq_length=a.max_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=7,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok, peft_config=lora, data_collator=collator)
    trainer.model.print_trainable_parameters()
    trainer.train()

    lora_dir = Path(a.out) / "lora"
    trainer.model.save_pretrained(lora_dir)
    tok.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")

    if a.no_merge:
        return
    # Merge the adapter into a full model so Ollama / llama.cpp can load it.
    del trainer, model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, lora_dir).merge_and_unload()
    merged_dir = Path(a.out) / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    (merged_dir / "axia_nota_train.json").write_text(json.dumps(vars(a), indent=2))
    print(f"Merged model saved to {merged_dir}  -> next: see SETUP.md step 14")


if __name__ == "__main__":
    main()
