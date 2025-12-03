# llm_model.py

import json
import re
from typing import List, Dict, Any

from unsloth import FastLanguageModel
from transformers import GenerationConfig
from unsloth.chat_templates import get_chat_template
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage

from config import LORA_ADAPTER_PATH, MODEL_BASE

# ==============================================================================
# 4. CARREGAMENTO DO MODELO
# ==============================================================================

print("Loading Model Base and LoRA Adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_ADAPTER_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    device_map = "auto",
)
tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-thinking")

# ==============================================================================
# 5. FUNÇÕES AUXILIARES
# ==============================================================================

def convert_messages_to_qwen_format(messages: List[BaseMessage]) -> List[Dict]:
    """Converte mensagens LangChain para formato Qwen3"""
    qwen_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Lógica para System Prompt (conforme o código original)
            if hasattr(msg, 'name') and msg.name == 'system':
                role = "system"
            else:
                role = "user"
            qwen_messages.append({"role": role, "content": msg.content})
            
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_calls = []
                for tool_call in msg.tool_calls:
                    # Garantir que tool_call seja um dicionário (LangChain Tool Call)
                    if isinstance(tool_call, dict) and 'name' in tool_call and 'args' in tool_call:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"])
                            }
                        })
                if tool_calls:
                    qwen_messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls
                    })
            elif msg.content:
                qwen_messages.append({"role": "assistant", "content": msg.content})
                
        elif isinstance(msg, ToolMessage):
            qwen_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
    
    return qwen_messages

def convert_tools_to_qwen_format(tools_list: List) -> List[Dict]:
    """Converte tools LangChain para formato Qwen3"""
    qwen_tools = []
    
    for tool in tools_list:
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
            except Exception as e:
                print(f"[WARNING] Erro ao extrair schema da tool {tool.name}: {e}")
        
        qwen_tools.append(tool_schema)
    
    return qwen_tools

def parse_tool_calls_from_response(response_text: str):
    """Parseia tool calls da resposta do Qwen3 - VERSÃO CORRIGIDA"""
    print(f"[PARSING] Analisando resposta...")
    
    # 🎯 PADRÃO MAIS PRECISO - apenas conteúdo dentro de <tool_call>
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
    
    if matches:
        print(f"[PARSING SUCCESS] Encontrados {len(matches)} tool calls brutos")
        tool_calls = []
        
        for i, match in enumerate(matches):
            try:
                json_str = match.strip()
                
                # 🎯 CRÍTICO: IGNORAR EXEMPLOS DE DOCUMENTAÇÃO
                if "<function-name>" in json_str or "<args-json-object>" in json_str:
                    print(f"[PARSING SKIP] Ignorando exemplo de documentação: {json_str[:50]}...")
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
                    print(f"[PARSING SUCCESS] Tool call {i}: {tool_name} com args: {arguments}")
                else:
                    print(f"[PARSING WARNING] Tool call inválida: nome={tool_name}, args={arguments}")
                    
            except json.JSONDecodeError as e:
                print(f"[PARSING ERROR] JSON inválido no tool call {i}: {e}")
                print(f"[PARSING ERROR] JSON string: {match[:100]}...")
            except Exception as e:
                print(f"[PARSING ERROR] Erro geral no tool call {i}: {e}")
        
        print(f"[PARSING FINAL] {len(tool_calls)} tool calls válidas após filtragem")
        return tool_calls if tool_calls else None
    
    print("[PARSING] Nenhum tool call encontrado no formato esperado")
    return None

def clean_llm_response(response_text: str, text_input: str) -> str:
    """Limpa a resposta bruta do LLM"""
    # 1. Remover o prompt original e o token de parada
    clean_text = response_text.replace(text_input, "").strip()

    # 2. Remover tokens de stop (como eos_token_id)
    if tokenizer.eos_token:
        clean_text = clean_text.replace(tokenizer.eos_token, "").strip()
        
    # 3. Remover tags de tool call restantes
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', clean_text, flags=re.DOTALL)
    
    # 4. Remover tags de pensamento
    clean_text = re.sub(r'<think>(.*?)</think>', r'\1', clean_text, flags=re.DOTALL).strip()
    
    return clean_text
    
def compress_context(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Comprime contexto mantendo apenas o essencial"""
    if len(messages) <= 3: # 🎯 Máximo 3 mensagens
        return messages
        
    print(f"📦 Comprimindo contexto: {len(messages)} -> 3 mensagens")
    
    compressed = []
    
    # 🎯 Sempre manter: system prompt (apenas uma vez)
    system_msg = next((msg for msg in messages if isinstance(msg, HumanMessage) and hasattr(msg, 'name') and msg.name == 'system'), None)
    if system_msg:
        compressed.append(system_msg)
    
    # 🎯 Última mensagem do usuário
    user_msgs = [msg for msg in messages if isinstance(msg, HumanMessage) and (not hasattr(msg, 'name') or msg.name != 'system')]
    if user_msgs:
        compressed.append(user_msgs[-1])
    
    # 🎯 Última interação (se houver)
    last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if last_ai_msg:
        compressed.append(last_ai_msg)
        
    return compressed