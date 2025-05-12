<h1 align="center">MLMathMentor</h1>

<h2 align="center">Used Libraries</h2>
<div align="center">
  
 <a href="https://www.python.org" target="_blank" rel="noreferrer" style="display: inline-block;"> 
   <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="60" height="60"/>
 </a>

 <a href="https://numpy.org/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original-wordmark.svg" title="Numpy" alt="Numpy" width="60" height="60"/> 
 </a>

 <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer" style="display: inline-block;"> 
   <img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original.svg" title="tensorflow" alt="tensorflow" width="60" height="60"> 
 </a>

 <a href="https://pytorch.org/" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/21003710?s=200&v=4" title="pytorch" alt="pytorch" width="60" height="60"/> 
 </a>

 <a href="https://keras.io/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/keras/keras-original.svg" title="keras" alt="keras" width="60" height="60"> 
 </a>
 <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer" style="display: inline-block;">
   <img src="https://github.com/devicons/devicon/blob/master/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="60" height="60"/> 
 </a>

 <a href="https://huggingface.co/docs/hub/index" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4" title="huggingface" alt="huggingface" width="60" height="60"/> 
 </a>

 <a href="https://github.com/facebookresearch/faiss" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/16943930?s=48&v=4" title="faiss" alt="faiss" width="60" height="60"/> 
 </a>

 <a href="https://github.com/bitsandbytes-foundation/bitsandbytes" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/175231607?s=48&v=4" title="bitsandbytes" alt="bitsandbytes" width="60" height="60"/> 
 </a>

 <a href="https://github.com/run-llama/llama_index" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/130722866?s=48&v=4" title="llama_index" alt="llama_index" width="60" height="60"/> 
 </a>

 <a href="https://github.com/phoenixframework/phoenix" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/6510388?s=48&v=4" title="phoenix" alt="phoenix" width="60" height="60"/> 
 </a>

<a href="https://github.com/NVIDIA/NeMo-Guardrails" target="_blank" rel="noreferrer" style="display: inline-block;">
    <img src="https://avatars.githubusercontent.com/u/1728152?s=48&v=4" title="NeMo-Guardrails" alt="NeMo-Guardrails" width="60" height="60"/> 
 </a>


</div>


## Installation of Dependencies

You can install the required dependencies either manually or using the `requirements.txt` file.

###  Install Manually
```bash
pip install load_dotenv
pip install faiss-cpu
pip install bitsandbytes
pip install llama_index
pip install langchain_huggingface
pip install llama-index-embeddings-langchain
pip install llama-index-vector-stores-faissv
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-huggingface
pip install -U langchain-community sentence-transformers
pip install pyvis
pip install arize-phoenix openinference-instrumentation-llama-index opentelemetry-sdk --quiet
pip install nemoguardrails

````

### Install using the `requirements.txt` file
```bash
pip install -r requirements.txt

````


<h2 align="center"> MLMathMentor — AI-бот-репетитор по линейной алгебре и машинному обучению</h2>

## 🌟 О проекте
Описание: 
Проект представляет собой RAG-систему (Retrieval-Augmented Generation), обученную на математических книгах по линейной алгебре, математическому анализу и статистике.
Модель: Saiga LLaMA3 8B
База знаний: VectorStoreIndex — векторная база, используемая для быстрого поиска релевантной информации по запросу пользователя.
Защита и контроль запросов:
Запросы проверяются с помощью NeMo Guardrails — система отслеживает, соответствует ли вопрос допустимой теме (например, линейной алгебре), и ограничивает обработку нежелательных запросов.
RAG-пайплайн:
Поиск ответов по векторной базе знаний.
Ранжирование найденных результатов с помощью LLMRerank для выбора наиболее релевантных фрагментов.
Трассировка всех запросов и ответов с помощью Phoenix для удобного мониторинга, отладки и аналитики.

🚀 Как работает:
Пользователь задает вопрос.

NeMo Guardrails проверяет, соответствует ли вопрос теме.

Если запрос допустим:

Запускается RAG-поиск по векторной базе знаний.

LLMRerank выбирает наиболее релевантные фрагменты.

Phoenix трассирует этот процесс.

LLM формирует финальный ответ с объяснениями и примерами.

Если вопрос не по теме — бот возвращает корректное уведомление.

Как запустить


🗃️ Структура проекта

```bash
my_api_project/
├── create_storage/
│   ├── KnowledgeGraphIndex/  
│   │   ├── KnowledgeGraphIndex/               # тестовый вариант KnowledgeGraphIndex (долгое время обработки)
│   │   │   ├── default__vector_store.json     # файлы сохраненного KnowledgeGraphIndex
│   │   │   ├── docstore.json                  # файлы сохраненного KnowledgeGraphIndex
│   │   │   ├── graph_store.json               # файлы сохраненного KnowledgeGraphIndex   
│   │   │   ├── image__vector_store.json       # файлы сохраненного KnowledgeGraphIndex
│   │   │   └── index_store.json               # файлы сохраненного KnowledgeGraphIndex
│   │   │
│   │   ├── KnowledgeGraphIndex.ipynb          # код подготовки базы знаний 
│   │   ├── graph.html                         # визуализация данных 
│   │   └── graph_temp.html                    # временная визуализация данных 
│   │
│   ├── VectorStoreIndex/
│   │   ├── VectorStoreIndex/                  # выбранный вариант, используется в main файле 
│   │   │   ├── default__vector_store.json     # файлы сохраненного VectorStoreIndex  
│   │   │   ├── docstore.json                  # файлы сохраненного VectorStoreIndex
│   │   │   ├── graph_store.json               # файлы сохраненного VectorStoreIndex
│   │   │   ├── image__vector_store.json       # файлы сохраненного VectorStoreIndex
│   │   │   └── index_store.json               # файлы сохраненного VectorStoreIndex
│   │   │
│   │   └── VectorStoreIndex.ipynb             # код подготовки базы знаний 
│   │ 
│   └── library/                               # книги, использованные для обучения базы знаний
│
├── model/
│   └── model.ipynb                            # настройка модели и переменные окружения
│
├── main.ipynb                                 # основной файл: сборка LLM и реализация RAG
├── config.yaml                                # конфигурация для NeMo Guardrails
├── colang_content.yaml                        # правила ограничений и допустимых тем для NeMo Guardrails
├── .env                                       # переменные окружения
├── .gitignore                                 # список файлов/папок для игнорирования в git
├── requirements.txt                           # зависимости проекта
└── README.md                                  # документация проекта

```





Модель была взята из <a href="https://huggingface.co/docs/hub/index">Hugging-Face</a> и использует
<a href = "https://huggingface.co/IlyaGusev/saiga_llama3_8b"> saiga_llama3_8b </a> как базову  