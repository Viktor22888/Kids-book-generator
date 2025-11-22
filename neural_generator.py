import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

class NeuralTextGenerator:
 
    def __init__(self, mode="lightweight"):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.api_key = None  
        
        if mode == "lightweight":
            self._load_lightweight_model()
        elif mode == "api":
            self._setup_api()
        elif mode == "local":
            self._load_local_model()
    
    def _load_lightweight_model(self):
      

        
        try:
 
            model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
            
            print("📥 Скачивание модели (это займет несколько минут при первом запуске)...")
            print("💡 После скачивания модель будет работать БЕЗ интернета и БЕСПЛАТНО!")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                resume_download=True  
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=False,
                resume_download=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ БЕСПЛАТНАЯ нейросеть загружена и готова к работе!")
            print("🎉 Теперь работает ОФЛАЙН и БЕСПЛАТНО!")
            return True
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"⏱️  Таймаут при скачивании: {error_msg[:100]}")
                print("💡 Попробуйте позже - модель большая (~300MB)")
                print("💡 Или используйте режим без нейросети (neural_mode=None)")
            else:
                print(f"❌ Ошибка загрузки модели: {error_msg[:200]}")
                print("💡 Попробуйте позже или используйте режим без нейросети")
            return False
    
    def _setup_api(self):
      
        print("🌐 Режим API: будет использоваться интернет для генерации")
        print("🇷🇺 Поддержка БЕСПЛАТНЫХ российских API:")
        print("   - YANDEX_API_KEY для Yandex GPT (бесплатный тариф)")
        print("   - GIGACHAT_API_KEY для GigaChat от Сбера (бесплатный тариф)")
        
   
        self.yandex_api_key = os.getenv("YANDEX_API_KEY")
    
        self.gigachat_auth_key = os.getenv("GIGACHAT_AUTH_KEY") 
        self.gigachat_client_id = os.getenv("GIGACHAT_CLIENT_ID")
        self.gigachat_client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
   
        self.gigachat_api_key = os.getenv("GIGACHAT_API_KEY") or self.gigachat_auth_key
        self.yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")  
        
      
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = self.yandex_api_key or self.gigachat_api_key or self.gigachat_auth_key or self.openai_api_key
        
        if not self.api_key:
            print("⚠️  API ключ не найден!")
            print("💡 Для БЕСПЛАТНОЙ работы установите:")
            print("   - YANDEX_API_KEY (бесплатный тариф Yandex GPT)")
            print("   - GIGACHAT_API_KEY (бесплатный тариф GigaChat)")
            print("💡 Или используйте neural_mode='lightweight' для локальной модели")
            return False
        

        if self.yandex_api_key:
            self.api_provider = "yandex"
            print("✅ Найден Yandex GPT API ключ (бесплатный тариф)")
        elif self.gigachat_auth_key or (self.gigachat_client_id and self.gigachat_client_secret):
            self.api_provider = "gigachat"
            if self.gigachat_auth_key:
                print("✅ Найден GigaChat Authorization Key (бесплатный тариф)")
            else:
                print("✅ Найден GigaChat Client ID/Secret (бесплатный тариф)")
        elif self.gigachat_api_key:
            self.api_provider = "gigachat"
            print("✅ Найден GigaChat API ключ (бесплатный тариф)")
        elif self.openai_api_key:
            self.api_provider = "openai"
            print("⚠️  Используется OpenAI API (платно)")
        else:
            self.api_provider = None
        
        return True
    
    def _load_local_model(self):

        print("📂 Поиск локальной модели...")

        local_path = os.getenv("LOCAL_MODEL_PATH", None)
        
        if local_path and os.path.exists(local_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                self.model = AutoModelForCausalLM.from_pretrained(local_path)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Локальная модель загружена!")
                return True
            except Exception as e:
                print(f"❌ Ошибка загрузки локальной модели: {e}")
        
        print("💡 Локальная модель не найдена. Используйте lightweight режим.")
        return False
    
    def generate_with_neural_network(self, prompt, max_length=200, temperature=0.8):
        """
        Генерация текста с использованием реальной нейросети
        """
        if self.mode == "lightweight" and self.model and self.tokenizer:
            return self._generate_local(prompt, max_length, temperature)
        elif self.mode == "api":
            return self._generate_api(prompt, max_length, temperature)
        elif self.mode == "local" and self.model and self.tokenizer:
            return self._generate_local(prompt, max_length, temperature)
        else:
            return None
    
    def _generate_local(self, prompt, max_length, temperature):
        """
        Генерация с использованием локальной модели
        """
        try:
            full_prompt = f"Детская история: {prompt}. Начнем рассказ:"
            
           
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
           
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
           
            new_text = generated_text[len(full_prompt):].strip()
            
            return new_text
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None
    
    def _generate_api(self, prompt, max_length, temperature):
        """
        Генерация через API (требует интернет)
        Поддерживает БЕСПЛАТНЫЕ российские API: Yandex GPT и GigaChat
        """
        try:
            import requests
            
     
            if self.api_provider == "yandex" and self.yandex_api_key:
                if not self.yandex_folder_id:
                    print("⚠️  YANDEX_FOLDER_ID не установлен. Нужен для Yandex GPT.")
                    return None
                
                url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
                headers = {
                    "Authorization": f"Api-Key {self.yandex_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "modelUri": f"gpt://{self.yandex_folder_id}/yandexgpt/latest",
                    "completionOptions": {
                        "stream": False,
                        "temperature": temperature,
                        "maxTokens": str(max_length)
                    },
                    "messages": [
                        {
                            "role": "system",
                            "text": "Ты талантливый детский писатель. Пиши простые, понятные истории для детей 5-10 лет на русском языке. Каждая страница должна содержать уникальное событие. Избегай повторений."
                        },
                        {
                            "role": "user",
                            "text": prompt
                        }
                    ]
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["result"]["alternatives"][0]["message"]["text"]
                else:
                    print(f"Ошибка Yandex API: {response.status_code} - {response.text}")
                    return None
            
        
            elif self.api_provider == "gigachat":
        
                auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
                
      
                if self.gigachat_auth_key:
       
                    import uuid
                    auth_headers = {
                        "Authorization": f"Basic {self.gigachat_auth_key}",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "RqUID": str(uuid.uuid4())  
                    }
                elif self.gigachat_client_id and self.gigachat_client_secret:
           
                    import base64
                    import uuid
                    credentials = f"{self.gigachat_client_id}:{self.gigachat_client_secret}"
                    encoded_credentials = base64.b64encode(credentials.encode()).decode()
                    auth_headers = {
                        "Authorization": f"Basic {encoded_credentials}",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "RqUID": str(uuid.uuid4())
                    }
                elif self.gigachat_api_key:
          
                    auth_headers = {
                        "Authorization": f"Bearer {self.gigachat_api_key}",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json"
                    }
                else:
                    print("⚠️  GigaChat ключи не найдены")
                    return None
                
                auth_data = {"scope": "GIGACHAT_API_PERS"}
                
                try:
                   
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    auth_response = requests.post(auth_url, headers=auth_headers, data=auth_data, timeout=10, verify=False)
                    if auth_response.status_code != 200:
                        print(f"Ошибка получения токена GigaChat: {auth_response.status_code}")
                        return None
                    
                    access_token = auth_response.json().get("access_token")
                    if not access_token:
                        print("Не удалось получить токен GigaChat")
                        return None
                    
                    api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
                    api_headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    api_data = {
                        "model": "GigaChat",
                        "messages": [
                            {
                                "role": "system", 
                                "content": "Ты талантливый детский писатель. Пиши простые, понятные истории для детей 5-10 лет на русском языке. Каждая страница должна содержать уникальное событие или развитие сюжета. Избегай повторений. Пиши ярко и интересно."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_length,
                        "temperature": temperature
                    }
                    
                    api_response = requests.post(api_url, headers=api_headers, json=api_data, timeout=30, verify=False)
                    
                    if api_response.status_code == 200:
                        result = api_response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        print(f"Ошибка GigaChat API: {api_response.status_code} - {api_response.text}")
                        return None
                except Exception as e:
                    print(f"Ошибка GigaChat: {e}")
                    return None
            
            elif self.api_provider == "openai" and self.openai_api_key:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "Ты детский писатель. Пиши простые, понятные истории для детей на русском языке."},
                        {"role": "user", "content": f"Напиши детскую историю о: {prompt}"}
                    ],
                    "max_tokens": max_length,
                    "temperature": temperature
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"Ошибка OpenAI API: {response.status_code}")
                    return None
            else:
                print("⚠️  API ключ не установлен")
                return None
        except Exception as e:
            print(f"Ошибка API генерации: {e}")
            return None


class GPT4AllGenerator:

    
    def __init__(self):
        try:
            from gpt4all import GPT4All
            print("🔄 Инициализация GPT4All...")
            self.model = GPT4All("ggml-gpt4all-j-v1.3-groovy")
            print("✅ GPT4All готов!")
        except ImportError:
            print("❌ GPT4All не установлен. Установите: pip install gpt4all")
            self.model = None
        except Exception as e:
            print(f"❌ Ошибка инициализации GPT4All: {e}")
            self.model = None
    
    def generate(self, prompt, max_tokens=200):
        if not self.model:
            return None
        
        try:
            full_prompt = f"Напиши детскую историю на русском языке о: {prompt}\n\nИстория:"
            response = self.model.generate(full_prompt, max_tokens=max_tokens, temp=0.8)
            return response
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None

