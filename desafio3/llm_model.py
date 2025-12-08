# llm_model.py

import json
import re
import logging
from typing import List, Dict, Any

from unsloth import FastLanguageModel
from transformers import GenerationConfig
from unsloth.chat_templates import get_chat_template
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage

from config import LORA_ADAPTER_PATH, MODEL_BASE
from logging_llm import setup_logger  # Importar o setup_logger

# ==============================================================================
# 1. CONFIGURAÇÃO DE LOGGING
# ==============================================================================

# Configurar logger para llm_model
setup_logger("llm_model", "llm_model.log", logging.DEBUG)
llm_logger = logging.getLogger("llm_model")

# ==============================================================================
# 2. CARREGAMENTO DO MODELO
# ==============================================================================

llm_logger.info("Iniciando carregamento do Model Base e LoRA Adapter...")
print("Loading Model Base and LoRA Adapter...")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_ADAPTER_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        device_map = "auto",
    )
    llm_logger.info(f"Modelo carregado com sucesso de: {LORA_ADAPTER_PATH}")
    llm_logger.info(f"Configuração do modelo: max_seq_length=2048, load_in_4bit=True")
    
    tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-thinking")
    llm_logger.info("Chat template 'qwen3-thinking' aplicado ao tokenizer")
    
except Exception as e:
    llm_logger.error(f"Erro ao carregar o modelo: {e}", exc_info=True)
    raise

# ==============================================================================
# 3. FUNÇÕES AUXILIARES
# ==============================================================================

def convert_messages_to_qwen_format(messages: List[BaseMessage]) -> List[Dict]:
    """Converte mensagens LangChain para formato Qwen3"""
    llm_logger.debug(f"Convertendo {len(messages)} mensagens para formato Qwen3")
    qwen_messages = []
    
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            # Lógica para System Prompt (conforme o código original)
            if hasattr(msg, 'name') and msg.name == 'system':
                role = "system"
                llm_logger.debug(f"Mensagem {i}: System prompt detectado")
            else:
                role = "user"
            qwen_messages.append({"role": role, "content": msg.content})
            llm_logger.debug(f"Mensagem {i}: HumanMessage como {role} - {msg.content[:50]}...")
            
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_calls = []
                llm_logger.debug(f"Mensagem {i}: AIMessage com {len(msg.tool_calls)} tool calls")
                for j, tool_call in enumerate(msg.tool_calls):
                    # Garantir que tool_call seja um dicionário (LangChain Tool Call)
                    if isinstance(tool_call, dict) and 'name' in tool_call and 'args' in tool_call:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"])
                            }
                        })
                        llm_logger.debug(f"  Tool call {j}: {tool_call['name']} - {tool_call['args']}")
                if tool_calls:
                    qwen_messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls
                    })
            elif msg.content:
                qwen_messages.append({"role": "assistant", "content": msg.content})
                llm_logger.debug(f"Mensagem {i}: AIMessage sem tool calls - {msg.content[:50]}...")
                
        elif isinstance(msg, ToolMessage):
            qwen_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
            llm_logger.debug(f"Mensagem {i}: ToolMessage para tool_call_id={msg.tool_call_id} - {msg.content[:50]}...")
    
    llm_logger.info(f"Conversão concluída: {len(qwen_messages)} mensagens no formato Qwen3")
    return qwen_messages

def convert_tools_to_qwen_format(tools_list: List) -> List[Dict]:
    """Converte tools LangChain para formato Qwen3"""
    llm_logger.info(f"Convertendo {len(tools_list)} ferramentas para formato Qwen3")
    qwen_tools = []
    
    for i, tool in enumerate(tools_list):
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
            }
        }
        
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                args_schema = tool.args_schema.model_json_schema()
                tool_schema["function"]["parameters"] = {
                    "type": args_schema.get("type", "object"),
                    "properties": args_schema.get("properties", {}),
                    "required": args_schema.get("required", [])
                }
                llm_logger.debug(f"Tool {i} ({tool.name}): Schema extraído com sucesso")
            except Exception as e:
                llm_logger.warning(f"Tool {i} ({tool.name}): Erro ao extrair schema: {e}")
        else:
            llm_logger.debug(f"Tool {i} ({tool.name}): Sem args_schema definido")
        
        qwen_tools.append(tool_schema)
    
    llm_logger.info(f"Conversão de tools concluída: {len(qwen_tools)} tools convertidas")
    return qwen_tools

def parse_tool_calls_from_response(response_text: str):
    """Parseia tool calls da resposta do Qwen3 - VERSÃO CORRIGIDA"""
    llm_logger.info("Iniciando parse de tool calls da resposta do LLM")
    llm_logger.debug(f"Resposta bruta (primeiros 500 chars): {response_text[:500]}...")
    
    # 🎯 PADRÃO MAIS PRECISO - apenas conteúdo dentro de <tool_call>
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
    
    if matches:
        llm_logger.info(f"Encontrados {len(matches)} tool calls brutos na resposta")
        tool_calls = []
        
        for i, match in enumerate(matches):
            try:
                json_str = match.strip()
                llm_logger.debug(f"Tool call {i} bruto: {json_str[:100]}...")
                
                # 🎯 CRÍTICO: IGNORAR EXEMPLOS DE DOCUMENTAÇÃO
                if "<function-name>" in json_str or "<args-json-object>" in json_str:
                    llm_logger.warning(f"Tool call {i} ignorado (exemplo de documentação): {json_str[:50]}...")
                    continue
                
                tool_data = json.loads(json_str)
                
                tool_name = tool_data.get('name')
                arguments = tool_data.get('arguments', {})
                
                if tool_name and isinstance(arguments, dict):
                    tool_calls.append({
                        "name": tool_name,
                        "args": arguments,
                        "id": f"{tool_name}_call_{i}"
                    })
                    llm_logger.info(f"Tool call {i} válido: {tool_name} com args: {arguments}")
                else:
                    llm_logger.warning(f"Tool call {i} inválido: nome={tool_name}, args={arguments}")
                    
            except json.JSONDecodeError as e:
                llm_logger.error(f"JSON inválido no tool call {i}: {e}")
                llm_logger.debug(f"String JSON problemática: {match[:100]}...")
            except Exception as e:
                llm_logger.error(f"Erro geral no tool call {i}: {e}", exc_info=True)
        
        llm_logger.info(f"Parse concluído: {len(tool_calls)} tool calls válidas após filtragem")
        return tool_calls if tool_calls else None
    
    llm_logger.info("Nenhum tool call encontrado no formato esperado")
    return None

def clean_llm_response(response_text: str, text_input: str) -> str:
    """Limpa a resposta bruta do LLM"""
    llm_logger.debug("Iniciando limpeza da resposta do LLM")
    
    # 1. Remover o prompt original e o token de parada
    clean_text = response_text.replace(text_input, "").strip()
    llm_logger.debug(f"Texto após remover prompt original: {len(clean_text)} caracteres")

    # 2. Remover tokens de stop (como eos_token_id)
    if tokenizer.eos_token:
        clean_text = clean_text.replace(tokenizer.eos_token, "").strip()
        llm_logger.debug("Token EOS removido")
        
    # 3. Remover tags de tool call restantes
    before_tool_clean = len(clean_text)
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', clean_text, flags=re.DOTALL)
    after_tool_clean = len(clean_text)
    if before_tool_clean != after_tool_clean:
        llm_logger.debug(f"Tool calls removidos: {before_tool_clean - after_tool_clean} caracteres")
    
    # 4. Remover tags de pensamento
    before_think_clean = len(clean_text)
    clean_text = re.sub(r'<think>(.*?)</think>', r'\1', clean_text, flags=re.DOTALL).strip()
    after_think_clean = len(clean_text)
    if before_think_clean != after_think_clean:
        llm_logger.debug(f"Tags think removidas: {before_think_clean - after_think_clean} caracteres")
    
    llm_logger.info(f"Limpeza concluída: {len(response_text)} -> {len(clean_text)} caracteres")
    llm_logger.debug(f"Texto limpo (primeiros 200 chars): {clean_text[:200]}...")
    
    return clean_text

def compress_context(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Comprime contexto mantendo apenas o essencial"""
    if len(messages) <= 3: # 🎯 Máximo 3 mensagens
        llm_logger.debug(f"Contexto já comprimido: {len(messages)} mensagens (limite: 3)")
        return messages
        
    llm_logger.info(f"Comprimindo contexto: {len(messages)} -> 3 mensagens")
    
    compressed = []
    
    # 🎯 Sempre manter: system prompt (apenas uma vez)
    system_msg = next((msg for msg in messages if isinstance(msg, HumanMessage) and hasattr(msg, 'name') and msg.name == 'system'), None)
    if system_msg:
        compressed.append(system_msg)
        llm_logger.debug("System prompt mantido na compressão")
    
    # 🎯 Última mensagem do usuário
    user_msgs = [msg for msg in messages if isinstance(msg, HumanMessage) and (not hasattr(msg, 'name') or msg.name != 'system')]
    if user_msgs:
        compressed.append(user_msgs[-1])
        llm_logger.debug(f"Última mensagem de usuário mantida: {user_msgs[-1].content[:50]}...")
    
    # 🎯 Última interação (se houver)
    last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if last_ai_msg:
        compressed.append(last_ai_msg)
        if last_ai_msg.tool_calls:
            llm_logger.debug(f"Última AIMessage mantida com {len(last_ai_msg.tool_calls)} tool calls")
        else:
            llm_logger.debug(f"Última AIMessage mantida: {last_ai_msg.content[:50]}...")
        
    llm_logger.info(f"Contexto comprimido com sucesso: {len(messages)} -> {len(compressed)} mensagens")
    return compressed