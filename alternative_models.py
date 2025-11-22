import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class AlternativeBookGenerator:
    def __init__(self, model_choice="russian_gpt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_choice = model_choice
        self.model = None
        self.tokenizer = None
        self.pipe = None

        if model_choice == "pipeline":
            self._load_pipeline_model()
        else:
            self._load_direct_model()
    
    def _load_pipeline_model(self):
        print("Загрузка модели через pipeline...")
        try:
            self.pipe = pipeline(
                "text-generation",
                model="ai-forever/rugpt3small_based_on_gpt2",
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print("✅ Модель загружена через pipeline")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            self.pipe = None
    
    def _load_direct_model(self):
        print("Загрузка модели напрямую...")
        try:
            if self.model_choice == "ruGPT3":
                model_name = "ai-forever/rugpt3large_based_on_gpt2"
            else:
                model_name = "ai-forever/rugpt3small_based_on_gpt2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Модель {model_name} загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_with_pipeline(self, prompt, max_length=200):
        if not self.pipe:
            return None
        try:
            full_prompt = f"Детская история: {prompt}. Начнем рассказ:"
            result = self.pipe(
                full_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
            )
            return result[0]['generated_text']
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None
    
    def generate_with_direct_model(self, prompt, max_length=200):
        if not self.model or not self.tokenizer:
            return None
        try:
            full_prompt = f"Детская история: {prompt}. Начнем рассказ:"
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(full_prompt):].strip()
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None
    
    def generate_text(self, prompt, max_length=200):
        if self.pipe:
            return self.generate_with_pipeline(prompt, max_length)
        else:
            return self.generate_with_direct_model(prompt, max_length)


class ChatModelGenerator:
    def __init__(self, model_name="ai-forever/rugpt3small_based_on_gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Модель {model_name} загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_chat(self, user_message, max_new_tokens=100):
        if not self.model or not self.tokenizer:
            return None
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": f"Напиши детскую историю о: {user_message}"}
                ]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        do_sample=True
                    )
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                return generated_text
            else:
                prompt = f"Детская история: {user_message}. Начнем рассказ:"
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=len(inputs[0]) + max_new_tokens,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text[len(prompt):].strip()
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None


if __name__ == "__main__":
    print("Пример 1: Использование pipeline")
    generator1 = AlternativeBookGenerator(model_choice="pipeline")
    text1 = generator1.generate_text("драконе", max_length=150)
    print(f"Сгенерированный текст: {text1}\n")
    
    print("Пример 2: Прямая загрузка модели")
    generator2 = AlternativeBookGenerator(model_choice="russian_gpt")
    text2 = generator2.generate_text("приключениях в лесу", max_length=150)
    print(f"Сгенерированный текст: {text2}\n")
    
    print("Пример 3: Использование chat template")
    chat_generator = ChatModelGenerator()
    text3 = chat_generator.generate_chat("волшебном замке", max_new_tokens=100)
    print(f"Сгенерированный текст: {text3}\n")