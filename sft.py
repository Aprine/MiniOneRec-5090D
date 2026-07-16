import ast
import json
import math
import os
import random
from collections import defaultdict
from functools import partial

import fire
import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as handle:
            self.indices = json.load(handle)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens


def _safe_description_text(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return " ".join(str(v) for v in parsed if v)
        except (ValueError, SyntaxError):
            pass
    return text


def _item_semantic_text(features, max_description_chars=384):
    fields = [
        str(features.get("title", "") or ""),
        str(features.get("brand", "") or ""),
        str(features.get("categories", "") or ""),
        _safe_description_text(features.get("description", ""))[:max_description_chars],
    ]
    return " ".join(field.strip() for field in fields if field and field.strip())


def semantic_aware_sid_init(
    model,
    tokenizer,
    sid_index_path,
    item_meta_path,
    new_tokens,
    original_vocab_size,
    max_items_per_token=64,
    max_text_tokens=96,
):
    if not sid_index_path or not item_meta_path:
        print("Skipping SID semantic init: sid_index_path or item_meta_path is empty")
        return
    if not os.path.exists(sid_index_path) or not os.path.exists(item_meta_path):
        print("Skipping SID semantic init: index/meta file not found")
        return

    with open(sid_index_path, "r") as handle:
        indices = json.load(handle)
    with open(item_meta_path, "r") as handle:
        item_meta = json.load(handle)

    sid_to_texts = defaultdict(list)
    for item_id, sid_tokens in indices.items():
        features = item_meta.get(str(item_id))
        if not features:
            continue
        text = _item_semantic_text(features)
        if not text:
            continue
        for sid_token in sid_tokens:
            if len(sid_to_texts[sid_token]) < max_items_per_token:
                sid_to_texts[sid_token].append(text)

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    initialized = 0
    missing_text = 0

    with torch.no_grad():
        for sid_token in new_tokens:
            token_id = tokenizer.convert_tokens_to_ids(sid_token)
            if token_id is None or token_id < original_vocab_size:
                continue

            token_vectors = []
            for text in sid_to_texts.get(sid_token, []):
                ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_text_tokens,
                ).input_ids
                ids = [idx for idx in ids if 0 <= idx < original_vocab_size]
                if not ids:
                    continue
                id_tensor = torch.tensor(ids, device=input_embedding.weight.device)
                token_vectors.append(input_embedding.weight[id_tensor].float().mean(dim=0))

            if not token_vectors:
                missing_text += 1
                continue

            vector = torch.stack(token_vectors, dim=0).mean(dim=0).to(input_embedding.weight.dtype)
            input_embedding.weight[token_id].copy_(vector)
            if (
                output_embedding is not None
                and output_embedding.weight.shape[0] == input_embedding.weight.shape[0]
                and output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr()
            ):
                output_embedding.weight[token_id].copy_(vector.to(output_embedding.weight.dtype))
            initialized += 1

    print(
        "Semantic-aware SID init finished: "
        f"initialized={initialized}, missing_text={missing_text}, strategy=item_text_mean"
    )


def enable_new_token_embedding_training(model, original_vocab_size, tokenizer, new_tokens):
    if not new_tokens:
        print("Warning: no new tokens available for trainable SID embedding rows")
        return

    def enable_rows(embedding, label):
        if embedding is None or not hasattr(embedding, "weight"):
            return False
        if embedding.weight.shape[0] <= original_vocab_size:
            return False
        embedding.weight.requires_grad = True

        def mask_grad(grad):
            grad[:original_vocab_size].zero_()
            return grad

        embedding.weight.register_hook(mask_grad)
        print(
            f"Unfrozen {label} rows for {len(new_tokens)} new tokens "
            f"(indices {original_vocab_size} to {len(tokenizer) - 1})"
        )
        return True

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    input_enabled = enable_rows(input_embedding, "input embedding")

    output_enabled = False
    if (
        output_embedding is not None
        and input_embedding is not None
        and output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr()
    ):
        output_enabled = enable_rows(output_embedding, "output embedding/lm_head")

    if not input_enabled and not output_enabled:
        print("Warning: SID token rows were not unfrozen; check tokenizer resizing")


def normalize_lora_target_modules(value):
    if isinstance(value, str):
        return [module.strip() for module in value.split(",") if module.strip()]
    if isinstance(value, (list, tuple, set)):
        modules = []
        for item in value:
            modules.extend(normalize_lora_target_modules(item))
        return modules
    if value is None:
        return []
    return [str(value).strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warm_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warm_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    sid_init_strategy: str = "none",
    sid_init_max_items_per_token: int = 64,
    sid_init_max_text_tokens: int = 96,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,v_proj,o_proj",
    save_intermediate_checkpoints: bool = True,
    load_best_model_at_end: bool = True,
):
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    print(category)
    category = category_dict[category]
    assert base_model, "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    original_vocab_size = len(tokenizer)
    new_tokens = []

    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split(".")[0],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

            if sid_init_strategy == "item_text_mean":
                semantic_aware_sid_init(
                    model=model,
                    tokenizer=tokenizer,
                    sid_index_path=sid_index_path,
                    item_meta_path=item_meta_path,
                    new_tokens=new_tokens,
                    original_vocab_size=original_vocab_size,
                    max_items_per_token=sid_init_max_items_per_token,
                    max_text_tokens=sid_init_max_text_tokens,
                )
            elif sid_init_strategy != "none":
                raise ValueError(f"Unsupported sid_init_strategy: {sid_init_strategy}")

    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            enable_new_token_embedding_training(model, original_vocab_size, tokenizer, new_tokens)
        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("use_lora=True requires `pip install peft`.") from exc

        target_modules = normalize_lora_target_modules(lora_target_modules)
        if not target_modules:
            raise ValueError("use_lora=True requires at least one lora_target_modules entry")
        print(f"LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if freeze_LLM and new_tokens:
            enable_new_token_embedding_training(model, original_vocab_size, tokenizer, new_tokens)
        model.print_trainable_parameters()

    train_datasets = []
    train_data1 = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    train_datasets.append(train_data1)
    train_data2 = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    train_datasets.append(train_data2)
    train_data3 = FusionSeqRecDataset(
        train_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    train_datasets.append(train_data3)

    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        _checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    should_load_best_model = load_best_model_at_end and save_intermediate_checkpoints
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if should_load_best_model else []

    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step,
            save_strategy="steps" if save_intermediate_checkpoints else "no",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=should_load_best_model,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    final_model = trainer.model
    if use_lora and hasattr(final_model, "merge_and_unload"):
        print("Merging LoRA adapters into base model for standalone evaluation checkpoint")
        final_model = final_model.merge_and_unload()
    final_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    fire.Fire(train)
