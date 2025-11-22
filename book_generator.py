import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import base64
import io
from datetime import datetime
import time
import random
import math


class BookGenerator:
    def __init__(self, use_models=False, neural_mode="lightweight"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_models = use_models
        self.neural_mode = neural_mode
        print(f"Используется устройство: {self.device}")

        self.model = None
        self.tokenizer = None
        self.image_pipe = None
        self.neural_generator = None

        if use_models:
            self.model_name = "ai-forever/rugpt3small_based_on_gpt2"
            print("Загрузка языковой модели...")
            self.model, self.tokenizer = self._load_text_model_with_retry()
            if not self.model:
                print("⚠️  Языковая модель не загружена. Используется улучшенная генерация без моделей.")
            else:
                print("✅ Языковая модель загружена")

            print("Загрузка модели генерации изображений...")
            self.image_pipe = self._load_image_model_with_retry()
            if not self.image_pipe:
                print("⚠️  Модель изображений не загружена. Используется улучшенная генерация изображений.")
            else:
                print("✅ Модель изображений загружена")
        else:
            if neural_mode in ["lightweight", "api", "local"]:
                try:
                    import sys
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from neural_generator import NeuralTextGenerator
                    print(f"🤖 Инициализация реальной нейросети (режим: {neural_mode})...")
                    self.neural_generator = NeuralTextGenerator(mode=neural_mode)

                    model_loaded = (
                        self.neural_generator.model is not None
                        and self.neural_generator.tokenizer is not None
                    )
                    api_available = (
                        neural_mode == "api"
                        and self.neural_generator.api_key is not None
                    )
                    if model_loaded or api_available:
                        print("✅ Реальная нейросеть готова к работе!")
                    else:
                        print("⚠️  Нейросеть не загружена из-за проблем с интернетом.")
                        print("✅ Но это не проблема! Приложение работает отлично и без нейросети!")
                        self.neural_generator = None
                except ImportError as e:
                    print(f"⚠️  Модуль neural_generator не найден: {e}")
                    self.neural_generator = None
                except Exception as e:
                    print(f"⚠️  Не удалось загрузить нейросеть: {e}")
                    self.neural_generator = None
            else:
                print("📚 Режим работы: Улучшенная генерация БЕЗ нейросети")
                print("💡 Все функции доступны, интернет не требуется!")

    def _load_text_model_with_retry(self, max_retries=2):
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Повторная попытка загрузки модели (попытка {attempt + 1}/{max_retries + 1})...")
                    time.sleep(5)

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    resume_download=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    resume_download=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                model.to(self.device)
                model.eval()
                print("✅ Языковая модель загружена успешно")
                return model, tokenizer
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  Ошибка загрузки (попытка {attempt + 1}): {str(e)[:100]}...")
                    print("Повторная попытка через 5 секунд...")
                else:
                    print(f"❌ Не удалось загрузить модель после {max_retries + 1} попыток")
        return None, None

    def _load_image_model_with_retry(self, max_retries=1):
        model_name = "runwayml/stable-diffusion-v1-5"
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Повторная попытка загрузки модели изображений (попытка {attempt + 1}/{max_retries + 1})...")
                    time.sleep(5)

                pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=False,
                    resume_download=True,
                )
                pipe = pipe.to(self.device)
                print("✅ Модель генерации изображений загружена успешно")
                return pipe
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  Ошибка загрузки модели изображений (попытка {attempt + 1}): {str(e)[:100]}...")
                else:
                    print(f"❌ Не удалось загрузить модель изображений после {max_retries + 1} попыток")
        return None

    def generate_text(self, prompt, max_length=500, num_pages=5):
        if self.neural_generator:
            return self._generate_with_neural_network(prompt, max_length, num_pages)
        if self.model and self.tokenizer:
            return self._generate_with_model(prompt, max_length, num_pages)
        return self._simple_text_generation(prompt, num_pages)

    def _generate_with_neural_network(self, prompt, max_length, num_pages):
        pages = []
        story_context = []
        for page_num in range(num_pages):
            try:
                if page_num == 0:
                    page_prompt = f"Напиши начало детской истории о {prompt}. Начни с введения главного героя и места действия. Пиши простым языком для детей. Одно-два предложения."
                elif page_num == num_pages - 1:
                    context_summary = '. '.join([p['text'] for p in pages[:2]])[:200]
                    page_prompt = f"Продолжи детскую историю о {prompt}. Предыдущие события: {context_summary}. Напиши красивую концовку истории, где все заканчивается хорошо. Одно-два предложения."
                else:
                    context_summary = '. '.join([p['text'] for p in pages[-1:]])[:150]
                    page_prompt = f"Продолжи детскую историю о {prompt}. Что было: {context_summary}. Что происходит дальше? Добавь новое событие или приключение. Одно-два предложения."

                generated_text = self.neural_generator.generate_with_neural_network(
                    page_prompt,
                    max_length=150,
                    temperature=0.9,
                )
                if generated_text and len(generated_text.strip()) > 20:
                    page_text = generated_text.strip()
                    page_text = page_text.replace('**', '').replace('*', '').replace(' ', '').strip()
                    if page_text.startswith('История') or page_text.startswith('Жил-был'):
                        sentences = page_text.split('.')
                        if len(sentences) > 1:
                            page_text = '. '.join(sentences[1:]).strip()
                    if prompt.lower() in page_text.lower() and page_text.lower().startswith(prompt.lower()):
                        page_text = page_text[len(prompt):].strip()
                    if not page_text.endswith(('.', '!', '?')):
                        page_text += '.'
                    if len(page_text) > 300:
                        sentences = page_text.split('.')
                        page_text = '. '.join(sentences[:2]) + '.'
                    if page_text and page_text not in [p['text'] for p in pages]:
                        pages.append({"page_number": page_num + 1, "text": page_text})
                        story_context.append(page_text)
                    else:
                        pages.append({
                            "page_number": page_num + 1,
                            "text": self._get_unique_page_text(prompt, page_num, num_pages, story_context)
                        })
                else:
                    pages.append({
                        "page_number": page_num + 1,
                        "text": self._get_unique_page_text(prompt, page_num, num_pages, story_context)
                    })
            except Exception as e:
                print(f"Ошибка генерации нейросетью для страницы {page_num + 1}: {e}")
                pages.append({
                    "page_number": page_num + 1,
                    "text": self._get_unique_page_text(prompt, page_num, num_pages, story_context)
                })
        return pages

    def _get_unique_page_text(self, prompt, page_num, total_pages, context):
        if page_num == 0:
            beginnings = [
                f"Однажды в мире Майнкрафта жил отважный герой по имени Стив. Он любил исследовать бескрайние просторы и строить удивительные сооружения.",
                f"В далеком блоковом мире жил мальчик по имени Стив. Каждый день он отправлялся в новые приключения, открывая тайны этого удивительного мира.",
                f"История началась, когда Стив проснулся в незнакомом месте. Вокруг него простирался огромный мир Майнкрафта, полный загадок и приключений.",
            ]
            return random.choice(beginnings)
        if page_num == total_pages - 1:
            endings = [
                f"Так закончилось удивительное путешествие Стива. Он многому научился и нашел новых друзей в мире Майнкрафта. Все были счастливы!",
                f"Стив вернулся домой с новыми знаниями и впечатлениями. Его приключение в Майнкрафте стало легендой, которую рассказывают до сих пор.",
                f"В конце концов, все закончилось хорошо. Стив понял, что дружба и смелость - это самое важное в любом приключении.",
            ]
            return random.choice(endings)
        developments = [
            f"Стив отправился в глубокую пещеру, где нашел редкие алмазы. Он был очень осторожен, чтобы не встретить враждебных мобов.",
            f"На пути Стив встретил дружелюбную деревню. Жители помогли ему и поделились едой. Стив был благодарен за помощь.",
            f"Стив построил красивый дом из дерева и камня. Он украсил его факелами и цветами, чтобы было уютно и светло.",
            f"Внезапно на Стива напали зомби! Но он был готов и отважно защищался своим мечом. В конце концов, он победил всех врагов.",
            f"Стив нашел заброшенный храм в джунглях. Внутри его ждали опасные ловушки, но и ценные сокровища. Он был очень осторожен.",
        ]
        return random.choice(developments)

    def _generate_with_model(self, prompt, max_length, num_pages):
        pages = []
        full_prompt = f"Детская история: {prompt}. Начнем рассказ:"
        for page_num in range(num_pages):
            try:
                use_chat = hasattr(self.tokenizer, 'apply_chat_template') and hasattr(self.tokenizer, 'chat_template')
                if use_chat:
                    messages = [
                        {"role": "user", "content": f"Напиши детскую историю о: {prompt}. Продолжи рассказ для страницы {page_num + 1}."}
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
                            max_new_tokens=max_length // num_pages,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=1.2,
                        )
                    new_text = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[-1]:],
                        skip_special_tokens=True,
                    ).strip()
                else:
                    inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_length=len(inputs[0]) + max_length // num_pages,
                            num_return_sequences=1,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_p=0.9,
                            repetition_penalty=1.2,
                        )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    new_text = generated_text[len(full_prompt):].strip()
                sentences = new_text.split('.')
                page_text = '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else new_text[:200]
                page_text = page_text.strip()
                if not page_text:
                    page_text = f"Продолжение истории о {prompt}."
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text[:300],
                })
                full_prompt = generated_text[:500]
            except Exception as e:
                print(f"Ошибка генерации текста для страницы {page_num + 1}: {e}")
                pages.append({
                    "page_number": page_num + 1,
                    "text": self._simple_text_generation(prompt, 1)[0]["text"],
                })
        return pages

    def _simple_text_generation(self, prompt, num_pages):
        pages = []
        beginnings = [
            f"Однажды в волшебном мире произошла удивительная история о {prompt}.",
            f"В далекой стране жил маленький герой, который очень любил {prompt}.",
            f"История началась, когда наш герой впервые встретил {prompt}.",
            f"В сказочном лесу жила дружная семья, которая знала всё о {prompt}.",
        ]
        developments = [
            f"Главный герой отправился в захватывающее путешествие, чтобы узнать больше о {prompt}.",
            f"В пути герой встретил много интересных персонажей, связанных с {prompt}.",
            f"Вместе с новыми друзьями герой узнал много удивительных фактов о {prompt}.",
            f"Приключения продолжались, и каждый день приносил новые открытия о {prompt}.",
            f"Герой понял, что {prompt} - это нечто особенное и волшебное.",
            f"Вместе они преодолели множество препятствий, связанных с {prompt}.",
        ]
        endings = [
            f"В конце концов, все закончилось хорошо. История о {prompt} стала любимой для многих детей.",
            f"Герой вернулся домой с новыми знаниями о {prompt} и поделился ими с друзьями.",
            f"Так закончилась удивительная история о {prompt}, которая учит нас быть добрыми и любознательными.",
            f"Все были счастливы, и история о {prompt} стала легендой, которую рассказывают до сих пор.",
        ]
        details = [
            " Они смеялись и играли вместе.",
            " Солнце светило ярко, и все были счастливы.",
            " Птицы пели красивые песни.",
            " Цветы распускались вокруг них.",
        ]

        for i in range(num_pages):
            if i == 0:
                text = random.choice(beginnings) + " Это было необычное приключение, которое запомнится навсегда."
            elif i == num_pages - 1:
                text = random.choice(endings) + " И все жили долго и счастливо!"
            else:
                text = random.choice(developments) + random.choice(details)
            pages.append({"page_number": i + 1, "text": text})
        return pages

    def generate_image(self, text_description, page_number, query=""):
        if not self.image_pipe or not self.use_models:
            return self._create_placeholder_image(page_number, query)
        try:
            image_prompt = f"детская иллюстрация, яркие цвета, мультяшный стиль, {text_description}, качественная детская книга"
            with torch.no_grad():
                image = self.image_pipe(
                    image_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Ошибка генерации изображения: {e}")
            return self._create_placeholder_image(page_number, text_description)

    def _create_placeholder_image(self, page_number=1, query=""):
        color_schemes = [
            [(255, 220, 200), (255, 180, 150)],
            [(200, 230, 255), (150, 200, 255)],
            [(255, 240, 200), (255, 220, 150)],
            [(220, 255, 220), (180, 255, 180)],
            [(255, 200, 220), (255, 170, 190)],
            [(240, 220, 255), (220, 190, 255)],
        ]
        color1, color2 = color_schemes[(page_number - 1) % len(color_schemes)]
        width, height = 512, 512
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        center_x, center_y = width // 2, height // 2
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)
        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                factor = min(dist / max_dist, 1.0)
                r = int(color1[0] * (1 - factor) + color2[0] * factor)
                g = int(color1[1] * (1 - factor) + color2[1] * factor)
                b = int(color1[2] * (1 - factor) + color2[2] * factor)
                if page_number % 2 == 0:
                    circle_dist = math.sqrt((x - width // 4) ** 2 + (y - height // 4) ** 2)
                    if 30 < circle_dist < 50:
                        r = min(255, r + 30)
                        g = min(255, g + 30)
                        b = min(255, b + 30)
                pixels[x, y] = (r, g, b)
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
            draw.text((20, 20), f"Стр. {page_number}", fill=(100, 100, 100, 180), font=font)
        except:
            pass
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def generate_book(self, query):
        print(f"Генерация книги для запроса: {query}")
        pages_text = self.generate_text(query, max_length=1000, num_pages=6)
        pages = []
        for page_data in pages_text:
            image = self.generate_image(f"{query}, {page_data['text'][:50]}", page_data['page_number'], query)
            pages.append({
                "page_number": page_data['page_number'],
                "text": page_data['text'],
                "image": image,
            })
        book = {
            "title": f"История о {query}",
            "query": query,
            "pages": pages,
            "generated_at": datetime.now().isoformat(),
            "total_pages": len(pages),
        }
        return book