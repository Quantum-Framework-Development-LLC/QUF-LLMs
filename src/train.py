from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_model():
    # Initialize the GPT-2 model and tokenizer
    model = GPT2LMHeadModel(GPT2Config())
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="./data/train.txt",
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    train_model()
