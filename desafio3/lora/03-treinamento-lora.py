# -*- coding: utf-8 -*-
"""Script de Fine-Tuning QLoRA (Unsloth + Qwen3)
Adaptado para usar dataset local e formato de dados já conversacional.
"""

# Importações principais
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
torch._dynamo.config.disable = True
import os, re
from transformers import TextStreamer, GenerationConfig

# --- CONFIGURAÇÃO DE CAMINHOS E MODELO ---
# ⚠️ Ajuste estes caminhos se for necessário!
LORA_ADAPTER_PATH = r"C:\Users\robso\Documents\FIAP-POS\fase3-fiap\tech-challenge-3\lora_model_qwen3_medquad"
MODEL_BASE = "unsloth/Qwen3-1.7B"
MEU_ARQUIVO_DATASET = r"C:\Users\robso\Documents\FIAP-POS\fase3-fiap\tech-challenge-3\dataset_medquad_fine_tuning.jsonl"


# ==============================================================================
# 🎯 CORREÇÃO CRÍTICA PARA WINDOWS (Erro de Multiprocessamento/RuntimeError)
# Todo o código principal deve ser envolvido neste bloco.
# ==============================================================================
if __name__ == '__main__':
    
    # --- CARREGAR MODELO E TOKENIZER ---
    print(f"Loading Model: {MODEL_BASE}")
    
    # 🎯 CORREÇÃO CRÍTICA PARA ERRO DE VRAM/DISPATCH
    # Adicionando device_map="auto" para garantir que o modelo seja carregado
    # corretamente na GPU, evitando dispatch para CPU ou disco.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_BASE,
        max_seq_length = 2048,
        load_in_4bit = True, # QLoRA: 4 bit quantization
        load_in_8bit = False,
        full_finetuning = False,
        device_map = "auto", 
    )

    # --- CONFIGURAR PEFT/LoRA ---
    print("Configuring LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # --- PREPARAÇÃO DOS DADOS (SIMPLIFICADA) ---

    # 1. Carregar Template de Chat (Qwen-3)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )

    # 2. 🎯 Carregar o seu dataset local
    print(f"[STATUS] Carregando dataset local: {MEU_ARQUIVO_DATASET}")
    
    # Adicionado num_proc=1 para desabilitar o multiprocessamento na tokenização,
    # que é a fonte do RuntimeError no Windows.
    dataset = load_dataset("json", data_files=MEU_ARQUIVO_DATASET, split = "train", num_proc=1)

    # 3. ⚠️ IMPORTANTE: Como seu dataset já parece estar no formato 'text' pronto,
    # pulamos as etapas de formatação, e o SFTTrainer usará a coluna "text".

    # --- TREINAR O MODELO ---
    print("\n" + "="*60)
    print("[STATUS] CONFIGURANDO TREINAMENTO LoRA...")
    print("="*60)

    # Cria o diretório de salvamento, se não existir
    os.makedirs(LORA_ADAPTER_PATH, exist_ok=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None,
        args = SFTConfig(
            output_dir = LORA_ADAPTER_PATH, # Onde logs e checkpoints serão salvos
            dataset_text_field = "text", # Usa a coluna 'text' pré-formatada
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, # Batch efetivo de 4
            warmup_steps = 5,
            max_steps = 60, # Definido para um teste rápido. Mude para None para treino completo!
            learning_rate = 2e-4,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
            fp16 = not torch.cuda.is_bf16_supported(), # Ajuste automático para tipo de precisão
            bf16 = torch.cuda.is_bf16_supported(),
        ),
    )

    # --- MÁSCARA PARA TREINAR APENAS NAS RESPOSTAS (Melhora a precisão) ---
    trainer = train_on_responses_only(
        trainer,
        # Padrões de início de turno (ajustados para o seu formato)
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    # 6. INICIAR TREINAMENTO!
    print("\n" + "="*60)
    print("[STATUS] INICIANDO TREINAMENTO LoRA...")
    print("="*60)

    trainer_stats = trainer.train()

    # --- SALVAR MODELO ---
    model.save_pretrained(LORA_ADAPTER_PATH) # Salvamento do adaptador LoRA
    tokenizer.save_pretrained(LORA_ADAPTER_PATH)
    print("\n[STATUS] Adaptador LoRA e Tokenizer salvos em:", LORA_ADAPTER_PATH)


    # --- INFERÊNCIA DE TESTE ---
    print("\n"+"="*60)
    print("TESTE DE INFERÊNCIA APÓS TREINAMENTO")
    print("="*60)

    # Exemplo de pergunta médica para testar o domínio treinado (MedQuAD)
    messages = [
        {"role" : "user", "content" : "O que é Leucemia Linfoblástica Aguda em Adultos (LLA) e quais são os seus principais sintomas? Responda em Português."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )

    # Geração de texto
    # Configuração de geração para melhor controle
    generation_config = GenerationConfig(
        max_new_tokens = 256,
        temperature = 0.7,
        top_p = 0.8,
        top_k = 20,
        do_sample = True,
        pad_token_id = tokenizer.eos_token_id, # Importante para Qwen
        eos_token_id = tokenizer.eos_token_id,
    )

    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        generation_config = generation_config,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    # --- AVALIAÇÃO DO TESTE ---
    print("\n"+"="*60)
    print("AVALIAÇÃO DO TESTE DE INFERÊNCIA")
    print("="*60)
    print("O teste de inferência (acima) deve mostrar uma resposta **direta e informativa** sobre a LLA, de preferência em **Português** (como solicitado no prompt), refletindo o conhecimento do dataset MedQuAD.")
    print("Se a resposta for coerente, precisa e no idioma correto, o fine-tuning foi bem-sucedido.")