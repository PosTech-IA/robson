# agent_graph.py

import operator
import time
import re
import json
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END

from config import SYSTEM_PROMPT, DB_NAME
from monitoring import MonitoringSystem
from tools import tool_map, tools
import psycopg2
from psycopg2 import extras
from transformers import GenerationConfig
from llm_model import (
    model, 
    tokenizer, 
    convert_messages_to_qwen_format, 
    convert_tools_to_qwen_format, 
    parse_tool_calls_from_response, 
    clean_llm_response,
    compress_context
)

# Inicializar sistema de monitoramento (Deve ser global/passado no grafo)
monitor = MonitoringSystem()

# ==============================================================================
# 3. DEFINIÇÃO DO ESTADO DO AGENTE COM MONITORING
# ==============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    monitoring_data: Dict[str, Any] 

# ==============================================================================
# 6. NÓS DO LANGGRAPH COM MONITORING
# ==============================================================================

def monitor_node(state: AgentState):
    """
    🆕 NÓ DE MONITORAMENTO - Coleta métricas e analytics
    """
    start_time = monitor.start_timer()
    
    print("\n" + "="*60)
    print("📈 MONITOR NODE - Analytics em Tempo Real")
    print("="*60)
    
    # Coletar métricas do estado atual
    last_message = state['messages'][-1] if state.get('messages') else None
    
    # Exibir analytics
    monitor.print_real_time_metrics("monitor", state)
    
    # Análise detalhada do estado
    if last_message:
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"🔧 Tool Calls Detectados: {len(last_message.tool_calls)}")
            for tc in last_message.tool_calls:
                print(f"   - {tc['name']}: {tc.get('args', {})}")
        elif isinstance(last_message, ToolMessage):
            print(f"📨 Última Tool Result: {last_message.content[:100]}...")
        elif isinstance(last_message, AIMessage):
            print(f"💬 Resposta do Assistente: {last_message.content[:100]}...")
    
    duration = time.time() - start_time
    monitor.log_node_execution("monitor", duration)
    
    print("="*60)
    return state

def call_model_with_tools(state: AgentState):
    """Nó 1: LLM (Qwen3) que decide qual tool usar, sem forçar ferramentas de consulta."""
    start_time = monitor.start_timer()
    
    print("="*40)
    print("🚀 NÓ: call_model_with_tools (LLM)")
    
    messages = state["messages"]
    
    # 🎯 COMPRIMIR CONTEXTO
    compressed_messages = compress_context(messages)
    print(f"📦 Contexto comprimido: {len(messages)} -> {len(compressed_messages)} mensagens")
    
    # Garantir system prompt
    system_prompt_found = any(
        isinstance(msg, HumanMessage) and hasattr(msg, 'name') and msg.name == 'system' 
        for msg in compressed_messages
    )
    
    if not system_prompt_found:
        compressed_messages.insert(0, HumanMessage(content=SYSTEM_PROMPT, name="system"))
    
    
    # -------------------------------------------------------------------------
    # ❌ Lógica de detecção antecipada (forçando SQL_query_tool) REMOVIDA.
    #    Agora, o LLM DEVE usar a ferramenta 'check_and_schedule_availability'
    #    quando o usuário solicita agendamento e fornece os IDs/data.
    # -------------------------------------------------------------------------

    
    # Converter formatos para Qwen3
    qwen_messages = convert_messages_to_qwen_format(compressed_messages)
    # CRÍTICO: 'tools' deve conter a nova tool 'check_and_schedule_availability'
    qwen_tools = convert_tools_to_qwen_format(tools)
    
    # Aplicar template
    text_input = tokenizer.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=qwen_tools,
    )
    
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=800,
        temperature=0.7, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_tokens = model.generate(**inputs, generation_config=generation_config)
    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    
    print(f"[LOG] Resposta bruta do LLM (primeiros 500 chars):\n{response_text[:500]}...")
    
    # Parse tool calls PRIMEIRO
    tool_calls = parse_tool_calls_from_response(response_text)
    
    if tool_calls:
        # 🎯 EXTRAIR THINKING MESMO COM TOOL CALLS
        think_content = ""
        if "<think>" in response_text and "</think>" in response_text:
            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                print(f"[LOG] 🧠 Thinking com tool call: {think_content[:100]}...")
        
        # 🎯 SE TEM THINKING, criar mensagem com thinking + tool calls
        if think_content:
            ai_message = AIMessage(
                content=f"🧠 **Raciocínio do Assistente:**\n\n{think_content}",
                tool_calls=tool_calls
            )
        else:
            ai_message = AIMessage(content="", tool_calls=tool_calls)
            
        print(f"[LOG] ✅ Tool Call detectado: {tool_calls[0]['name']}")
        result = {"messages": [ai_message]}
    else:
        # 🎯 SEM TOOL CALLS: limpeza normal
        clean_response = clean_llm_response(response_text, text_input)
        print(f"[LOG] Resposta limpa: {clean_response[:100]}...")
        result = {"messages": [AIMessage(content=clean_response)]}
    
    duration = time.time() - start_time
    monitor.log_node_execution("call_model", duration)
    
    return result
def generate_natural_response(tool_results: List[Dict], state: AgentState) -> str:
    """Gera resposta natural baseada nos resultados das tools - VERSÃO CORRIGIDA"""
    
    user_query = ""
    # Achar a última mensagem de usuário para contexto
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and (not hasattr(msg, 'name') or msg.name != 'system'):
            user_query = msg.content.lower()
            break
    
    tool_result = tool_results[0]
    tool_name = tool_result["name"]
    tool_output = tool_result["output"]
    
    print(f"🧠 Gerando resposta para: '{user_query}'")
    
    try:
        data = json.loads(tool_output)
        
        if tool_name == "SQL_query_tool":
            # 🎯 DETECÇÃO MAIS INTELIGENTE BASEADA NOS DADOS
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                
                # 🎯 SE TEM 'nome' E 'nome_especialidade' → É LISTA DE MÉDICOS
                if 'nome' in first_item and 'nome_especialidade' in first_item:
                    medicos = [f"**{item['nome']}** - {item['nome_especialidade']}" for item in data]
                    return f"👨‍⚕️ **Corpo Médico do Hospital {DB_NAME}:**\n\n" + "\n".join(medicos) + f"\n\nTotal de {len(medicos)} profissionais em nossa equipe."
                
                # 🎯 SE TEM APENAS 'nome_especialidade' → É LISTA DE ESPECIALIDADES
                elif 'nome_especialidade' in first_item and 'nome' not in first_item:
                    especialidades = [item['nome_especialidade'] for item in data]
                    return f"🎯 **Especialidades Médicas Disponíveis:**\n\nNo hospital {DB_NAME}, temos {len(especialidades)} especialidades:\n\n• " + "\n• ".join(especialidades) + f"\n\nEstas são todas as especialidades do nosso corpo clínico."
                
                # 🎯 SE TEM 'nome' APENAS → É LISTA DE NOMES
                elif 'nome' in first_item:
                    nomes = [item['nome'] for item in data]
                    return f"📋 **Registros Encontrados ({len(data)}):**\n\n• " + "\n• ".join(nomes)
            
            # Resposta para lista vazia
            elif isinstance(data, list) and len(data) == 0:
                return "🔍 **Resultado da Consulta:**\n\nNão encontrei registros correspondentes à sua pesquisa."
            
            # Resposta genérica
            else:
                return f"📊 **Consulta Realizada:**\n\nProcessei sua solicitação e obtive {len(data) if isinstance(data, list) else 1} registro(s)."
        
        elif tool_name == "check_and_schedule_availability":
            # O output já é um JSON
            agendamento_data = data
            if agendamento_data.get("status") == "agendado_sucesso":
                return f"✅ **Agendamento Confirmado!**\n\n• Médico: {agendamento_data.get('medico', 'N/A')}\n• Especialidade: {agendamento_data.get('especialidade', 'N/A')}\n• Data/Hora: {agendamento_data.get('data', 'N/A')}\n\n{agendamento_data.get('mensagem', 'Agendamento realizado com sucesso!')}"
            else:
                return f"❌ **Falha no Agendamento:**\n\n{agendamento_data.get('mensagem', 'Não foi possível completar o agendamento.')}"
        
        # Fallback genérico para JSON que não se encaixa nas regras
        return f"🤖 **Resposta Final:**\n\nConsegui processar sua solicitação e os dados obtidos foram:\n{tool_output[:500]}..."

    except json.JSONDecodeError:
        # Se a tool retornou algo que não é JSON (ex: erro de conexão do DB)
        return f"⚠️ **Alerta do Sistema:**\n\nA ferramenta retornou um erro inesperado:\n\n{tool_output}"
    except Exception as e:
        return f"⚠️ **Erro na Geração da Resposta:**\n\nHouve um erro ao formatar os dados de resposta: {e}"


def execute_tools(state: AgentState):
    """Nó 2: Executa tools e GERA RESPOSTA NATURAL"""
    start_time = monitor.start_timer()
    
    print("="*40)
    print("🛠️ NÓ: execute_tools (Tool Execution + Resposta Natural)")
    
    ai_message = state["messages"][-1]
    
    # 🎯 PRESERVAR O THINKING DA MENSAGEM ORIGINAL
    # A mensagem pode vir com conteúdo como: "🧠 **Raciocínio do Assistente:**..."
    original_thinking = ai_message.content if ai_message.content else ""
    
    if not ai_message.tool_calls:
        print("[LOG ERRO] Última mensagem não continha Tool Calls.")
        duration = time.time() - start_time
        monitor.log_node_execution("execute_tools", duration, success=False)
        return {"messages": [AIMessage(content="Erro: Tool call não encontrado.")]}

    success = True
    tool_results = []
    tool_messages = []
    
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"[LOG] Executando ferramenta: {tool_name}")
        print(f"[LOG] Argumentos: {tool_args}")
            
        if tool_name in tool_map:
            tool_func = tool_map[tool_name]
            try:
                # CRÍTICO: Usa tool_func.invoke(tool_args) onde tool_args já é um dicionário
                tool_output = tool_func.invoke(tool_args) 
                monitor.log_tool_call(tool_name)
                print(f"[LOG] ✅ Resultado da Tool: {tool_output[:200]}...")

                tool_results.append({
                    "name": tool_name,
                    "args": tool_args,
                    "output": tool_output
                })

                tool_messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )
            except Exception as e:
                success = False
                error_msg = f"Erro ao executar tool {tool_name}: {str(e)}"
                print(f"[LOG ERROR] {error_msg}")
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )
        else:
            success = False
            print(f"[LOG ERRO] Tool não encontrada: {tool_name}")
            tool_messages.append(
                ToolMessage(
                    content=f"Erro: Tool '{tool_name}' não encontrada.",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )
    
    duration = time.time() - start_time
    monitor.log_node_execution("execute_tools", duration, success=success)
    
    # 🎯 CRÍTICO: GERAR RESPOSTA NATURAL PRESERVANDO THINKING
    # Se houver resultados ou sucesso, tenta gerar uma resposta natural
    if tool_results and success:
        resposta_natural = generate_natural_response(tool_results, state)
        
        # 🎯 COMBINAR THINKING ORIGINAL COM RESPOSTA NATURAL
        final_response = ""
        # Verifica se o thinking original existe e o anexa
        if original_thinking and "🧠" in original_thinking:
            final_response = f"{original_thinking}\n\n---\n\n{resposta_natural}"
        else:
            final_response = resposta_natural
            
        print(f"🎯 RESPOSTA FINAL COM THINKING: {final_response[:150]}...")
        return {"messages": [AIMessage(content=final_response)]}
        
    else:
        # Se falhou, retorna mensagem de erro ou as tool_messages para o LLM re-analisar
        # Aqui, vamos retornar uma mensagem de erro final para evitar loop infinito
        error_msg = "Ocorreu um erro ao processar sua solicitação ou a ferramenta falhou."
        if original_thinking and "🧠" in original_thinking:
            error_msg = f"{original_thinking}\n\n---\n\n{error_msg}"
        
        # Se o LLM precisa de mais uma chance, retorne tool_messages para o call_model_with_tools.
        # Caso contrário, retorne AIMessage de erro final.
        return {"messages": tool_messages} # Retorna ToolMessages para o LLM (call_model) tentar novamente.


def route_tools(state: AgentState) -> str:
    """Roteador para decidir o próximo passo"""
    
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"[ROUTER] 🛠️ Tool calls detectados. Próximo nó: execute_tools")
        return "execute_tools"
    
    if isinstance(last_message, ToolMessage):
        print(f"[ROUTER] 📨 Resultado de Tool detectado. Próximo nó: call_model_with_tools (Re-análise)")
        return "call_model_with_tools"
    
    print(f"[ROUTER] 💬 Resposta Final detectada. Próximo nó: END")
    return "end"

# ==============================================================================
# 7. CONSTRUÇÃO DO GRAFO
# ==============================================================================

def build_agent_graph():
    """Constrói e compila o grafo do agente"""
    workflow = StateGraph(AgentState)

    # 1. Adicionar nós
    workflow.add_node("monitor_start", monitor_node)
    workflow.add_node("call_model_with_tools", call_model_with_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("monitor_end", monitor_node)

    # 2. Definir entrada
    workflow.set_entry_point("monitor_start")

    # 3. Definir edges (conexões)
    
    # Monitor (Início) -> LLM
    workflow.add_edge("monitor_start", "call_model_with_tools")
    
    # LLM -> Roteador
    workflow.add_conditional_edges(
        "call_model_with_tools",
        route_tools,
        {
            "execute_tools": "execute_tools",
            "end": "monitor_end"
        }
    )
    
    # Execução de Tools -> Roteador
    workflow.add_conditional_edges(
        "execute_tools",
        route_tools,
        {
            "call_model_with_tools": "call_model_with_tools", # Tool Message -> LLM re-analisa
            "end": "monitor_end"
        }
    )

    # Monitor (Fim) -> END
    workflow.add_edge("monitor_end", END)

    # 4. Compilar
    app = workflow.compile()
    print("\n✅ LangGraph compilado com sucesso.")
    return app

# A função build_agent_graph será chamada no main.py