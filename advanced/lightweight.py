import json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
import wandb
import time
import psutil
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# WandB Initialization
wandb.init(project="therapist-chatbot", name="gpt-neo-1.3B-quantized-lora-checkpointed")

# Load corpus.json data
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Prepare input-output pairs
data_pairs = []
for i in range(0, len(corpus)-1, 2):
    if corpus[i]['role'] == 'user' and corpus[i+1]['role'] == 'therapist':
        input_text = corpus[i]['content']
        output_text = corpus[i + 1]['content']
        data_pairs.append({"input": input_text, "output": output_text})

# Convert dataset to Hugging Face format
train_data = data_pairs
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# Load model and tokenizer with 4-bit quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Set pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Disable `use_cache` for gradient checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Dynamically Identify `torch.nn.Linear` layers for LoRA
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split('.')
        target_modules.add(names[0] if len(names) == 1 else names[-1])

# Remove unsupported layers if present in LoRA config
target_modules.discard("lm_head")
target_modules = list(target_modules)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules
)
model = get_peft_model(model, lora_config)

# Ensure all floating-point parameters require gradients
for param in model.parameters():
    if param.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:  # Only set for float types
        param.requires_grad = True


# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]
    inputs["labels"] = labels
    return inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training configuration with smaller batch size
training_args = TrainingArguments(
    output_dir="./results",
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=2,  # Reduced batch size for lower memory usage
    gradient_accumulation_steps=4,  # Accumulates gradients to simulate larger batch sizes
    num_train_epochs=10,
    save_total_limit=1,
    fp16=False,
    report_to="wandb"
)

# Custom Trainer with memory tracking
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_memory_usage_list = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # Runtime and memory tracking
        start_time = time.time()
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)
        gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            self.gpu_memory_usage_list.append(gpu_memory_usage)
        
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        # Logging runtime and memory usage
        end_time = time.time()
        runtime = end_time - start_time
        wandb.log({
            "train/loss": loss.item(),
            "train/runtime": runtime,
            "train/memory_usage_MB": memory_usage,
            "train/gpu_memory_usage_MB": gpu_memory_usage,
        })
        
        return (loss, outputs) if return_outputs else loss

    def log_average_gpu_memory_usage(self):
        if self.gpu_memory_usage_list:
            avg_gpu_memory_usage = sum(self.gpu_memory_usage_list) / len(self.gpu_memory_usage_list)
            wandb.log({"train/average_gpu_memory_usage_MB": avg_gpu_memory_usage})
            print(f"Average GPU Memory Usage: {avg_gpu_memory_usage:.2f} MB")

# Measure total training time
overall_start_time = time.time()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

train_result = trainer.train()
overall_end_time = time.time()
total_training_time = (overall_end_time - overall_start_time) / 60
wandb.log({"train/total_training_time_min": total_training_time})
print(f"Total Training Time: {total_training_time:.2f} minutes")

# Log average GPU memory usage
trainer.log_average_gpu_memory_usage()
trainer.save_model("./fine_tuned_therapist_chatbot")

df = pd.DataFrame(trainer.state.log_history)
print(df)
wandb.finish()
