import os
from transformers import (AutoTokenizer, 
                          AutoModelForMaskedLM, 
                          LineByLineTextDataset, 
                          DataCollatorForLanguageModeling, 
                          Trainer, 
                          TrainingArguments)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./TAPT_data.csv",
        block_size=512,
    )       
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./TAPT_models/",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=32,
        evaluation_strategy = 'steps',
        save_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,    
    )

    trainer.train()
    trainer.save_model("./TAPT_pretrained_model")