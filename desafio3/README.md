1. rodar para instalar dependencias
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Após, rodar para instalar os abaixo, pois o unsloth ja instala várias dependencias
Necessário baixar o cuda toolkit
https://developer.nvidia.com/cuda-toolkit

# --- LangChain e Componentes Core ---
# Instala o pacote principal do LangChain Agent e o módulo LLM/Pipeline
langchain
langchain-core
psycopg2
# O LangChain precisa do pacote 'langchain-community' para o HuggingFacePipeline
langchain-community
ddgs

langchain_ollama
IPython
Graphviz
langchain_huggingface
pydot

# subir banco com dados
docker-compose up -d para subir o postgres

# rodar o clonar-repo-trata-dados.py
baixa o repo e extrai os xml

# rodar converte-xml-vector.py
converte os XML em vetores jsonl

# rodar treinamento-lora.py
ele irá treinar o LLm com os dados dos vetores jsonl

# rodar teste-llm-pos-lora.py
após sucesso no treino  será criado o adaptador lora para usar nesse script