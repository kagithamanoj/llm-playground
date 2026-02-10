"""
LoRA Fine-Tuning Template — Fine-tune any Hugging Face model with LoRA/QLoRA.
Lightweight, memory-efficient, and production-ready.

Usage:
    python fine-tuning/lora_finetune.py --model "meta-llama/Llama-3.2-1B" --dataset "tatsu-lab/alpaca"
"""

import os
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FinetuneConfig:
    """Configuration for LoRA fine-tuning."""
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Auto-detected if None
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    # Quantization
    use_4bit: bool = True  # QLoRA
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    # Output
    output_dir: str = "./outputs/lora-model"


def get_quantization_config(config: FinetuneConfig):
    """Create BitsAndBytes quantization config for QLoRA."""
    import torch
    from transformers import BitsAndBytesConfig

    if not config.use_4bit:
        return None

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(config: FinetuneConfig):
    """Create LoRA configuration."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_model_and_tokenizer(config: FinetuneConfig):
    """Load base model with optional quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_config = get_quantization_config(config)
    token = os.getenv("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 512):
    """Load and tokenize dataset."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train[:5000]")  # Use subset for demo

    def format_prompt(example):
        """Format into instruction-following template."""
        if "instruction" in example and "output" in example:
            # Alpaca-style format
            text = f"### Instruction:\n{example['instruction']}\n\n"
            if example.get("input"):
                text += f"### Input:\n{example['input']}\n\n"
            text += f"### Response:\n{example['output']}"
        elif "text" in example:
            text = example["text"]
        else:
            text = str(example)
        return {"text": text}

    dataset = dataset.map(format_prompt)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    return dataset


def train(config: FinetuneConfig, dataset_name: str = "tatsu-lab/alpaca"):
    """Run the full fine-tuning pipeline."""
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    print(f"🚀 Fine-tuning {config.model_name} with LoRA")
    print(f"   LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"   QLoRA 4-bit: {config.use_4bit}")

    # Load model
    print("\n📥 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Prepare for training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    dataset = prepare_dataset(dataset_name, tokenizer, config.max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n🏋️ Training...")
    trainer.train()

    # Save
    print(f"\n💾 Saving adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("✅ Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", type=str, default="./outputs/lora-model")
    args = parser.parse_args()

    config = FinetuneConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        output_dir=args.output,
    )

    train(config, dataset_name=args.dataset)
