from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

lora_r: int = 8
lora_dropout: float = 0.1
lora_alpha: int = 32

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train[:17%]")
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # needed for 16-bit
    target_modules.remove("lm_head")

target_modules = list(target_modules)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

for lora_r in [8, 128, 256]:
    torch.cuda.empty_cache()
    wandb.init(project="Hanghae99", name=f"rank {lora_r}", group="lora")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    print("Max Alloc:", round(torch.cuda.max_memory_allocated(0) / 1024**3, 1), "GB")

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/tmp/clm-instruction-tuning",
            max_seq_length=128,
            per_device_train_batch_size=16,
            fp16=True,
            logging_steps=1,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    print(
        "Max Alloc After:", round(torch.cuda.max_memory_allocated(0) / 1024**3, 1), "GB"
    )

    trainer.train()

    wandb.finish()