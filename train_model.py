from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

def prepare_dataset(texts):
    full_text = "\n\n".join(texts)
    
    dataset = Dataset.from_dict({"text": [full_text]})
    
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def train_model(model_name="ai-forever/rugpt3small_based_on_gpt2", 
                train_texts=None,
                output_dir="./models/fine-tuned"):

    print("Загрузка модели и токенизатора...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if train_texts is None:
  
        train_texts = [
            "Жил-был маленький дракончик. Он любил играть с детьми и рассказывать им сказки.",
            "В далеком лесу жила дружная семья зверей. Они всегда помогали друг другу.",
            "Принцесса отправилась в путешествие, чтобы найти волшебный цветок.",
        ]
    
    print("Подготовка датасета...")
    dataset = prepare_dataset(train_texts)
    
 
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
    )
    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print("Начало обучения...")
    trainer.train()
    
    print(f"Сохранение модели в {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Обучение завершено!")

if __name__ == "__main__":

    
    print("""
    ВНИМАНИЕ: Это пример скрипта для обучения модели.
    
    Для реального обучения вам потребуется:
    1. Большой датасет русских детских книг (тысячи текстов)
    2. GPU с достаточным объемом памяти (рекомендуется 16GB+)
    3. Время на обучение (может занять часы или дни)
    
    Пример использования:
    python train_model.py
    """)


