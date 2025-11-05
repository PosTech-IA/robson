import os
import glob
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from tqdm import tqdm
import git
from datasets import Dataset
import codecs
import sys
try:
    if sys.stdout.encoding.lower() not in ['utf-8', 'cp65001']:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
except Exception:
    pass
# --- CONFIGURAÇÕES DO REPOSITÓRIO E CAMINHOS ---
REPO_URL = "https://github.com/abachaa/MedQuAD.git"
REPO_DIR = "MedQuAD_Data"
# Caminho interno onde os XMLs estão dentro do repo clonado
# A estrutura do MedQuAD é REPO_DIR/XML_files/...

# --- ETAPA 1.1: CLONAGEM DO REPOSITÓRIO (ROBUSTO) ---

def clone_repo_if_not_exists(url: str, dest_dir: str):
    """Clona o repositório se ele ainda não existir no caminho."""
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir):
        print(f"[STATUS] Diretório '{dest_dir}' já existe. Pulando clonagem.")
        return
    try:
        print(f"[STATUS] Clonando repositório de: {url}")
        git.Repo.clone_from(url, dest_dir)
        print("[STATUS] Clonagem concluída com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha ao clonar o repositório: {e}")
        # É importante levantar o erro se a clonagem falhar e os arquivos não existirem
        raise FileNotFoundError(f"Não foi possível obter os dados do GitHub. Erro: {e}")

# --- ETAPA 1.2: PARSER DE XML PERSONALIZADO (ADAPTAÇÃO AO MEDQUAD) ---

def parse_xml_file(file_path: str) -> List[Dict[str, str]]:
    """
    Extrai pares de Pergunta (Question) e Resposta (Answer) do XML do MedQuAD.
    
    A estrutura XML do MedQuAD é geralmente: <QA_SET><QA><Q>...<A>...</A></Q></QA></QA_SET>
    """
    extracted_data = []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Iterar sobre todos os blocos de QA (Question-Answer)
        # Assumindo a tag <QA> como o elemento que contém o par
        for qa_pair in root.findall('.//QAPair'):
            question_element = qa_pair.find('Question')
            answer_element = qa_pair.find('Answer')
            
            question = question_element.text.strip() if question_element is not None and question_element.text else ""
            answer = answer_element.text.strip() if answer_element is not None and answer_element.text else ""
            
            if question and answer:
                extracted_data.append({
                    "pergunta": question,
                    "resposta": answer,
                    "source": os.path.basename(file_path)
                })
                
    except ET.ParseError:
        # print(f"[AVISO] Ignorando arquivo XML malformado: {file_path}")
        pass
    except Exception as e:
        print(f"[ERRO INESPERADO] em {file_path}: {e}")
        
    return extracted_data

# --- ETAPA 1.3: ROTINA DE EXTRAÇÃO GLOBAL RECURSIVA ---

def read_xml_data_recursively(root_path: str) -> List[Dict[str, str]]:
    """Busca recursivamente todos os XMLs em subpastas e extrai o conteúdo."""
    
    # A busca recursiva com 'glob' e '**' é perfeita para subpastas
    search_path = os.path.join(root_path, '**', '*.xml')
    all_xml_files = glob.glob(search_path, recursive=True)
    
    if not all_xml_files:
        print(f"[AVISO] Nenhum arquivo XML encontrado em '{root_path}'.")
        return []

    print(f"[STATUS] Encontrados {len(all_xml_files)} arquivos XML para processar.")
    
    all_data = []
    for file_path in tqdm(all_xml_files, desc="Processando XMLs"):
        data = parse_xml_file(file_path)
        all_data.extend(data)
        
    return all_data

# --- ETAPA 1.4: LIMPEZA, CURADORIA E FORMATAÇÃO (LGPD/FT) ---

def clean_curate_and_format(dados_brutos: List[Dict[str, str]]) -> Dataset:
    """Aplica limpeza e formatação final para o Fine-Tuning."""
    
    # Conjunto para rastrear duplicatas e aplicar desduplicação
    unique_q_a = set()
    dataset_para_ft = []
    
    for item in tqdm(dados_brutos, desc="Curadoria e Formatação"):
        pergunta = item['pergunta']
        resposta = item['resposta']
        
        # --- (PASSO A): LIMPEZA DE TEXTO (essencial para LLMs) ---
        # Remove tags HTML residuais, caracteres especiais e normaliza espaços
        pergunta_limpa = ' '.join(pergunta.split())
        resposta_limpa = ' '.join(resposta.split())

        # --- (PASSO B): ANONIMIZAÇÃO (Placeholder) ---
        # ⚠️ IMPLEMENTAR AQUI A ROTINA DE ANONIMIZAÇÃO REAL (NER/Regex) ⚠️
        # Ex:
        # pergunta_anon = anonimizar_texto(pergunta_limpa) 
        # resposta_anon = anonimizar_texto(resposta_limpa)
        pergunta_anon = pergunta_limpa
        resposta_anon = resposta_limpa
        
        # --- (PASSO C): CURADORIA E DESDUPLICAÇÃO ---
        # Filtra dados de baixa qualidade
        if len(resposta_anon) < 20 or len(pergunta_anon) < 5:
            continue
            
        # Desduplicação: cria uma tupla imutável para verificar unicidade
        qa_pair = (pergunta_anon, resposta_anon)
        if qa_pair in unique_q_a:
            continue
        unique_q_a.add(qa_pair)
        
        # --- (PASSO D): FORMATAÇÃO FINAL PARA FINE-TUNING (Instruction Format) ---
        documento_final = (
            f"### Pergunta: {pergunta_anon}\n"
            f"### Resposta: {resposta_anon}"
        )
        
        dataset_para_ft.append({"text": documento_final})

    # Converte para o objeto Dataset do Hugging Face para uso direto no SFTTrainer
    print(f"[STATUS] Documentos únicos prontos para FT: {len(dataset_para_ft)}")
    return Dataset.from_list(dataset_para_ft)

# =========================================================================
# === FLUXO DE EXECUÇÃO PRINCIPAL ===
# =========================================================================

# 1. Obter os dados (Clonar ou verificar)
clone_repo_if_not_exists(REPO_URL, REPO_DIR)

# 2. Extrair os dados brutos de todas as subpastas XML
REPO_DIR = "MedQuAD_Data"
dados_brutos = read_xml_data_recursively(REPO_DIR)
print(f"[STATUS] Total de pares Pergunta-Resposta extraídos: {dados_brutos}")
if not dados_brutos:
    print("\n[FALHA] Nenhum dado extraído. Verifique o caminho e o parser XML.")
else:
    # 3. Limpar, Curar e Formatar
    hf_dataset = clean_curate_and_format(dados_brutos)
    
    # 4. Salvar o Dataset para o Fine-Tuning
    hf_dataset.to_json("dataset_medquad_ft.jsonl", orient="records", lines=True, force_ascii=False)
    
    print("\n[SUCESSO] Dataset finalizado. Prossiga para a Fase de Fine-Tuning (Treinamento LoRA).")