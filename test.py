from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
# Dummy reward function for demonstration purposes
def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]
training_args = GRPOConfig(
        output_dir=f"_vanilla",
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=0.001,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        # eval_strategy="steps",
        # eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        logging_steps=10,
        bf16=True,
        gradient_accumulation_steps=3,
        learning_rate=1e-6,
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_tensor_parallel_size=2,
        seed=3407,
        max_prompt_length=2048,
        max_completion_length=1024,
        num_generations=8,
        gradient_checkpointing=True,
        run_name="vanilla_" + f"_lr{1e-6}_kl{1e-3}",
        report_to = "wandb"
    )
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_letters,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()