# from llama_index.llms.huggingface import HuggingFaceLLM

# # from huggingface_hub import login
# # from dotenv import load_dotenv
# # import openai
# # import os
# # import torch
# from torch import bfloat16
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig, pipeline
# # import bitsandbytes
# # from llama_index.core import SimpleDirectoryReader
# # from llama_index.embeddings.langchain import LangchainEmbedding
# # from llama_index.core import Settings
# # from llama_index.core.graph_stores import SimpleGraphStore
# # from llama_index.core import KnowledgeGraphIndex
# # from llama_index.core import StorageContext
# # from langchain_huggingface  import HuggingFaceEmbeddings
# from llama_index.llms.huggingface import HuggingFaceLLM


# MODEL_NAME = "IlyaGusev/saiga_llama3_8b"

# DEFAULT_SYSTEM_PROMPT = """Ты — MLTeacherBot, интеллектуальный помощник и учитель математики для машинного обучения.
# Твоя задача — объяснять сложные математические темы понятным языком, помогать решать задачи и давать полезные советы.

# ### Твой стиль общения:
# - Дружелюбный, терпеливый, мотивирующий.
# - Объясняй просто и структурированно, подстраиваясь под уровень пользователя.
# - Используй аналогии и примеры.

# Ты — наставник, который делает обучение увлекательным и доступным.
# """

# # 4-разрядная конфигурация для загрузки LLM с меньшим объемом памяти графического процессора
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Использую 4-битную квантизацию
#     bnb_4bit_quant_type='nf4',  # используем формат NF4
#     bnb_4bit_use_double_quant=True,  # применить повторное квантовани
#     bnb_4bit_compute_dtype=bfloat16  # Тип из которого преобразуем (доступная точность модели)
# )

# # Загружаем модель с квантизацией и автоматическим распределением
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# model.eval()

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# print(generation_config)


# def messages_to_prompt(messages):
#     prompt = ""
    
#     for message in messages:
#         if message.role == 'system':
#             prompt += f"<s>{message.role}\n{message.content}</s>\n"
#         elif message.role == 'user':
#             prompt += f"<s>{message.role}\n{message.content}</s>\n"
#         elif message.role == 'bot':
#             prompt += f"<s>bot\n"


#     # ensure we start with a system prompt, insert blank if needed
#     if not prompt.startswith("<s>system\n"):
#         prompt = "<s>system\n</s>\n" + prompt

#     # add final assistant prompt
#     prompt = prompt + "<s>bot\n"
#     return prompt

# def completion_to_prompt(completion):
#     return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"


# llm = HuggingFaceLLM(
#     model=model,             # модель
#     model_name=MODEL_NAME,   # идентификатор модели
#     tokenizer=tokenizer,     # токенизатор
#     max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
#     model_kwargs={"quantization_config": quantization_config}, # параметры квантования
#     generate_kwargs = {   # параметры для инференса
#       "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
#       "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
#       "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
#       "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
#       "repetition_penalty": generation_config.repetition_penalty,
#       "temperature": generation_config.temperature,
#       "do_sample": True,
#       "top_k": generation_config.top_k,
#       "top_p": generation_config.top_p
#     },
#     messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
#     completion_to_prompt=completion_to_prompt, # функции для генерации текста
#     device_map="auto",                         # автоматически определять устройство
# )


# print("it's okay")
# from huggingface_hub.inference._types import ConversationalOutput

