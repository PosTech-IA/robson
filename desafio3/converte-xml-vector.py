import os
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from pprint import pprint

# --- CONFIGURAÇÕES ---
DATASET_PATH = "dataset_medquad_ft.jsonl" # O arquivo gerado na Fase 1
VECTOR_DB_PATH = "medquad_faiss_index"    # Pasta onde o Vector Store será salvo
# Modelo de Embedding: Recomendado para o português/multilíngue
# ou um modelo especializado em biomedicina, mas este é um bom ponto de partida:
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
# Alternativa Bio: "all-mpnet-base-v2"

# --- 1. FUNÇÃO METADATA (OPCIONAL, MAS RECOMENDADA) ---
# Adiciona metadados úteis (como o source original)
def metadata_func(record: dict, metadata: dict) -> dict:
    # O nosso 'text' contém a pergunta e resposta, mas a estrutura original
    # do JSONL deve ter o 'source' se você o incluiu na Fase 1.
    # Se o JSONL foi criado apenas com o campo 'text', este passo é ignorado.
    # Se o JSONL contém o dict {"text": "...", "source": "..."}:
    
    # Tentativa de extrair a pergunta para metadados (para melhor contexto)
    try:
        # Pega a pergunta formatada (após '### Pergunta: ' e antes de '\n### Resposta')
        pergunta = record['text'].split('### Pergunta: ')[-1].split('\n### Resposta')[0].strip()
        metadata["pergunta_original"] = pergunta
    except:
        metadata["pergunta_original"] = "N/A"
        
    return metadata

# --- 2. CARREGAR DOCUMENTOS (JSONL Loader) ---

print(f"[STATUS] Carregando documentos de: {DATASET_PATH}")

# O JSONLoader do LangChain é ideal para arquivos JSONL
# jq_schema='.' significa que cada linha (objeto JSON) é um documento
# content_key='text' diz qual campo deve ser usado como o conteúdo principal do documento
loader = JSONLoader(
    file_path=DATASET_PATH,
    jq_schema='.',
    content_key='text',
    json_lines=True,
    metadata_func=metadata_func # Adiciona metadados se necessário
)

# Carrega todos os documentos para a memória
documents = loader.load()
print(f"[STATUS] Total de documentos carregados: {len(documents)}")

# --- 3. SPLIT (DIVISÃO EM CHUNKS) ---
# O RAG funciona melhor com pedaços pequenos de informação
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # Tamanho do pedaço (em caracteres)
    chunk_overlap=50,     # Sobreposição para manter o contexto
    separators=["\n\n", "\n", ".", "!", "?", " "] # Prioridade de quebra
)

print("[STATUS] Dividindo documentos em chunks...")
chunks = text_splitter.split_documents(documents)
print(f"[STATUS] Total de chunks criados: {len(chunks)}")

# --- 4. EMBEDDING (CRIAÇÃO DOS VETORES) ---
print(f"[STATUS] Inicializando o modelo de embedding: {EMBEDDING_MODEL_NAME}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 5. CRIAÇÃO E SALVAMENTO DO VECTOR STORE (FAISS) ---
print("[STATUS] Criando o Vector Store FAISS...")

# Cria o índice FAISS a partir dos chunks e do modelo de embedding
vectorstore = FAISS.from_documents(chunks, embeddings)

# Salva o índice no disco para uso posterior (não precisa recriar)
vectorstore.save_local(VECTOR_DB_PATH)

print(f"\n[SUCESSO] Vector Store FAISS criado e salvo em: ./{VECTOR_DB_PATH}")
print("\n--- EXEMPLO DE BUSCA (Teste) ---")
# Teste de recuperação (Retrieval)
query = "Quais são os principais fatores de risco para Leucemia Linfocítica Crônica?"
docs_retrieved = vectorstore.similarity_search(query, k=2)

print(f"\nBusca por: '{query}'")
for i, doc in enumerate(docs_retrieved):
    print(f"\n--- Resultado {i+1} (Score: {doc.metadata.get('score', 'N/A')}) ---")
    print(doc.page_content[:200] + "...")