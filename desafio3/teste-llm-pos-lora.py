# -*- coding: utf-8 -*-
"""Script de Deploy e Inferência com LangGraph - COM MONITOR NODE"""

# Importações de Sistema e DB
import os
import json
import operator
import re
import time
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any

# Importações Unsloth/Transformers
from unsloth import FastLanguageModel
from transformers import GenerationConfig
from unsloth.chat_templates import get_chat_template

# Importações LangChain/LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Importações PostgreSQL
import psycopg2 
from psycopg2 import extras 

# ==============================================================================
# 0. CONFIGURAÇÕES GLOBAIS
# ==============================================================================

LORA_ADAPTER_PATH = r"C:\Users\robso\Documents\FIAP-POS\fase3-fiap\tech-challenge-3\lora_model_qwen3_medquad"
MODEL_BASE = "unsloth/Qwen3-1.7B"

DB_CONFIG = {
    'dbname': 'atividade3-fiap',
    'user': 'user',
    'password': 'password',
    'host': 'localhost',
    'port': '5432'
}
DB_NAME = DB_CONFIG['dbname']

# SCHEMA CORRETO DO BANCO
DATABASE_SCHEMA_INFO = """
SCHEMA DO BANCO atividade3-fiap:

TABELAS DISPONÍVEIS (USE MAIÚSCULAS):
- PACIENTES: paciente_id, nome, data_nascimento, cpf, telefone, endereco
- ESPECIALIDADES: especialidade_id, nome_especialidade
- MEDICOS: medico_id, nome, crm, especialidade_id, telefone
- CONSULTAS: consulta_id, paciente_id, medico_id, data_hora, tipo_atendimento, valor
- PRONTUARIOS: prontuario_id, consulta_id, diagnostico, historico_doenca, medicamentos_prescritos, observacoes

RELACIONAMENTOS:
- MEDICOS.especialidade_id → ESPECIALIDADES.especialidade_id
- CONSULTAS.paciente_id → PACIENTES.paciente_id
- CONSULTAS.medico_id → MEDICOS.medico_id
- PRONTUARIOS.consulta_id → CONSULTAS.consulta_id

"""

SYSTEM_PROMPT = f"""Você é o Dr. IA, um agente de saúde especializado no hospital {DB_NAME}, Você deve pensar cada etapa necessária antes de responder . 

{DATABASE_SCHEMA_INFO}

SUAS TAREFAS:
1. Para buscar dados (ex: 'Quais especialidades?', 'Liste os médicos'), use a ferramenta 'SQL_query_tool' com consultas SELECT válidas.
2. Para agendar consultas (ex: 'Gostaria de agendar...'), use a ferramenta 'schedule_appointment'.
3. Use APENAS os nomes de tabelas e colunas fornecidos acima, em MAIÚSCULAS.
4. APÓS executar uma ferramenta e receber o resultado, analise os dados e forneça uma resposta FINAL em português.

REGRAS IMPORTANTES:
- Use apenas as tabelas: PACIENTES, ESPECIALIDADES, MEDICOS, CONSULTAS, PRONTUARIOS
- Sempre use JOINs quando precisar relacionar tabelas
- Para especialidades, use a tabela ESPECIALIDADES
- Para agendamentos, use nomes de especialidades existentes
"""

# ==============================================================================
# 1. MONITORING SYSTEM - NOVO
# ==============================================================================

class MonitoringSystem:
    """Sistema de monitoramento para rastrear métricas do agente"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.metrics = {
            "session_start": datetime.now().isoformat(),
            "total_requests": 0,
            "tool_calls": 0,
            "sql_queries": 0,
            "errors": 0,
            "response_times": [],
            "node_executions": {}
        }
    
    def start_timer(self):
        return time.time()
    
    def log_node_execution(self, node_name: str, duration: float, success: bool = True):
        """Registra execução de nó com métricas"""
        if node_name not in self.metrics["node_executions"]:
            self.metrics["node_executions"][node_name] = {
                "count": 0,
                "total_time": 0,
                "errors": 0
            }
        
        self.metrics["node_executions"][node_name]["count"] += 1
        self.metrics["node_executions"][node_name]["total_time"] += duration
        
        if not success:
            self.metrics["node_executions"][node_name]["errors"] += 1
            self.metrics["errors"] += 1
    
    def log_tool_call(self, tool_name: str):
        """Registra chamada de tool"""
        self.metrics["tool_calls"] += 1
        if tool_name == "SQL_query_tool":
            self.metrics["sql_queries"] += 1
    
    def get_metrics_summary(self):
        """Retorna resumo das métricas"""
        total_time = sum(node["total_time"] for node in self.metrics["node_executions"].values())
        avg_time_per_node = {
            node: data["total_time"] / data["count"] 
            for node, data in self.metrics["node_executions"].items()
        }
        
        return {
            "session_id": self.session_id,
            "duration_seconds": round(total_time, 2),
            "total_requests": self.metrics["total_requests"],
            "tool_calls": self.metrics["tool_calls"],
            "sql_queries": self.metrics["sql_queries"],
            "errors": self.metrics["errors"],
            "node_performance": avg_time_per_node,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_real_time_metrics(self, current_node: str, state: Dict):
        """Exibe métricas em tempo real"""
        print(f"\n📊 [MONITOR] Nó: {current_node}")
        print(f"📋 Session: {self.session_id}")
        print(f"🔄 Total Requests: {self.metrics['total_requests']}")
        print(f"🛠️ Tool Calls: {self.metrics['tool_calls']}")
        print(f"🗄️ SQL Queries: {self.metrics['sql_queries']}")
        print(f"❌ Errors: {self.metrics['errors']}")
        
        # Informações do estado atual
        if 'messages' in state and state['messages']:
            last_msg = state['messages'][-1]
            msg_type = type(last_msg).__name__
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                print(f"🔧 Tool Calls Pendentes: {[tc['name'] for tc in last_msg.tool_calls]}")
            elif isinstance(last_msg, ToolMessage):
                print(f"📨 Tool Result: {last_msg.content[:80]}...")

# Inicializar sistema de monitoramento
monitor = MonitoringSystem()

# ==============================================================================
# 2. FUNÇÕES DO BANCO DE DADOS
# ==============================================================================

def execute_sql_query_impl(query: str) -> str:
    """Implementação da execução SQL com validação"""
    print("\n" + "~"*50)
    print(f"[EXECUÇÃO DA TOOL] Iniciando execução de Query SQL.")
    print(f"[QUERY BRUTA] {query}")
    
    # Validação básica da query
    query_lower = query.lower().strip()
    
    if not query_lower.startswith('select'):
        error_msg = "ERRO: Apenas consultas SELECT são permitidas."
        print(f"[DB ERRO] {error_msg}")
        return json.dumps({"status": "erro_seguranca", "mensagem": error_msg}, ensure_ascii=False)
    
    # Verificar tabelas válidas
    valid_tables = ['PACIENTES', 'ESPECIALIDADES', 'MEDICOS', 'CONSULTAS', 'PRONTUARIOS']
    table_used = None
    for table in valid_tables:
        if table.lower() in query_lower:
            table_used = table
            break
    
    if not table_used:
        error_msg = f"ERRO: Tabela não encontrada. Use apenas: {', '.join(valid_tables)}"
        print(f"[DB ERRO] {error_msg}")
        return json.dumps({"status": "erro_tabela", "mensagem": error_msg}, ensure_ascii=False)
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[DB STATUS] Conexão estabelecida com sucesso.")
        
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            print("[DB STATUS] Executando comando SQL...")
            cur.execute(query)
            
            result = cur.fetchall()
            print(f"[DB STATUS] SELECT concluído. Linhas retornadas: {len(result)}")
            
            if len(result) > 0:
                print(f"[DB LOG SAMPLE] Primeira linha: {dict(result[0])}")
                
            return json.dumps(result, ensure_ascii=False, default=str)

    except psycopg2.OperationalError as e:
        print(f"[DB ERRO FATAL] Falha de CONEXÃO: {e}")
        return json.dumps({"status": "erro_conexao", "mensagem": f"Erro de conexão: {e}"}, ensure_ascii=False)
        
    except psycopg2.Error as e:
        print(f"[DB ERRO SQL] Falha de EXECUÇÃO SQL: {e}")
        if conn:
            conn.rollback()
        
        error_suggestion = ""
        if "relation" in str(e) and "does not exist" in str(e):
            error_suggestion = f" Use apenas as tabelas em MAIÚSCULAS: {', '.join(valid_tables)}."
            
        return json.dumps({
            "status": "erro_sql", 
            "mensagem": f"Erro de SQL: {e}",
            "sugestao": error_suggestion
        }, ensure_ascii=False)
        
    finally:
        if conn:
            conn.close()
            print("[DB STATUS] Conexão com DB fechada.")
        print("~"*50)

@tool
def SQL_query_tool(query: str) -> str:
    """Executa uma consulta SELECT SQL no banco de dados"""
    return execute_sql_query_impl(query)

@tool
def schedule_appointment(doctor_name: str, date_time: str) -> str:
    """Simula o agendamento de uma consulta"""
    print(f"[EXECUÇÃO DA TOOL] Simulação de agendamento: {doctor_name} em {date_time}")
    
    # Validar se a especialidade existe
    validation_query = f"SELECT nome_especialidade FROM ESPECIALIDADES WHERE nome_especialidade ILIKE '%{doctor_name}%'"
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(validation_query)
            especialidades = cur.fetchall()
            
            if especialidades:
                especialidade_nome = especialidades[0]['nome_especialidade']
                medicos_query = f"SELECT m.nome FROM MEDICOS m JOIN ESPECIALIDADES e ON m.especialidade_id = e.especialidade_id WHERE e.nome_especialidade = '{especialidade_nome}'"
                cur.execute(medicos_query)
                medicos = cur.fetchall()
    except Exception as e:
        especialidades = []
        medicos = []
    finally:
        if conn:
            conn.close()
    
    if especialidades:
        especialidade_info = especialidades[0]['nome_especialidade']
        medico_nome = medicos[0]['nome'] if medicos else "Médico da especialidade"
        
        return json.dumps({
            "status": "agendado_sucesso",
            "medico": medico_nome,
            "especialidade": especialidade_info,
            "data": date_time,
            "mensagem": f"Agendamento confirmado com {medico_nome} ({especialidade_info}) para {date_time}"
        })
    else:
        return json.dumps({
            "status": "erro_agendamento", 
            "mensagem": f"Especialidade '{doctor_name}' não encontrada. Especialidades disponíveis: Cardiologia, Pediatria, Dermatologia, Clínica Geral, Ortopedia."
        })

# Lista de Tools
tools = [SQL_query_tool, schedule_appointment]
tool_map = {tool.name: tool for tool in tools}

# ==============================================================================
# 3. DEFINIÇÃO DO ESTADO DO AGENTE COM MONITORING
# ==============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    monitoring_data: Dict[str, Any]  # Novo campo para dados de monitoramento

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
            if hasattr(msg, 'name') and msg.name == 'system':
                role = "system"
            else:
                role = "user"
            qwen_messages.append({"role": role, "content": msg.content})
            
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_calls = []
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["args"])
                        }
                    })
                qwen_messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls
                })
            else:
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
    messages_count = len(state.get('messages', []))
    last_message = state['messages'][-1] if state.get('messages') else None
    
    # Exibir analytics
    monitor.print_real_time_metrics("monitor", state)
    
    # Análise detalhada do estado
    if last_message:
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"🔧 Tool Calls Detectados: {len(last_message.tool_calls)}")
            for tc in last_message.tool_calls:
                print(f"   - {tc['name']}: {tc.get('args', {})}")
        elif isinstance(last_message, ToolMessage):
            print(f"📨 Última Tool Result: {last_message.content[:100]}...")
        elif isinstance(last_message, AIMessage):
            print(f"💬 Resposta do Assistente: {last_message.content[:100]}...")
    
    duration = time.time() - start_time
    monitor.log_node_execution("monitor", duration)
    
    print("="*60)
    return state

def call_model_with_tools(state: AgentState):
    """Nó 1: LLM com detecção automática de agendamento"""
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
    
    # 🎯 DETECTAR INTENÇÃO DO USUÁRIO ANTES DE CHAMAR O MODELO
    user_query = ""
    for msg in compressed_messages:
        if isinstance(msg, HumanMessage) and (not hasattr(msg, 'name') or msg.name != 'system'):
            user_query = msg.content.lower()
            break
    
    # 🎯 SE É AGENDAMENTO, FORÇAR BUSCA DE MÉDICOS
    if any(word in user_query for word in ['agendar', 'marcar', 'consulta', 'agendamento', 'marcação']):
        print(f"🎯 DETECTADO: Solicitação de agendamento - '{user_query}'")
        
        # Detectar especialidade mencionada
        especialidade = None
        if "cardiologista" in user_query or "cardiologia" in user_query:
            especialidade = "Cardiologia"
        elif "pediatra" in user_query or "pediatria" in user_query:
            especialidade = "Pediatria" 
        elif "dermatologista" in user_query or "dermatologia" in user_query:
            especialidade = "Dermatologia"
        elif "ortopedia" in user_query or "ortopedista" in user_query:
            especialidade = "Ortopedia"
        elif "clínica geral" in user_query:
            especialidade = "Clínica Geral"
        
        if especialidade:
            print(f"🎯 Especialidade detectada: {especialidade}")
            # Forçar tool call para buscar médicos da especialidade
            forced_tool_calls = [{
                "name": "SQL_query_tool",
                "args": {
                    "query": f"SELECT nome, crm FROM MEDICOS WHERE especialidade_id = (SELECT especialidade_id FROM ESPECIALIDADES WHERE nome_especialidade = '{especialidade}')"
                },
                "id": f"forced_{especialidade.lower()}_call"
            }]
            print(f"🎯 Forçando busca de médicos de {especialidade}")
            duration = time.time() - start_time
            monitor.log_node_execution("call_model", duration)
            return {"messages": [AIMessage(content="", tool_calls=forced_tool_calls)]}
        else:
            # Se não detectou especialidade específica, forçar busca de todas especialidades
            forced_tool_calls = [{
                "name": "SQL_query_tool", 
                "args": {"query": "SELECT nome_especialidade FROM ESPECIALIDADES"},
                "id": "forced_especialidades_call"
            }]
            print("🎯 Forçando busca de especialidades disponíveis")
            duration = time.time() - start_time
            monitor.log_node_execution("call_model", duration)
            return {"messages": [AIMessage(content="", tool_calls=forced_tool_calls)]}
    
    # Converter formatos (apenas se não for agendamento)
    qwen_messages = convert_messages_to_qwen_format(compressed_messages)
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
def compress_context(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Comprime contexto mantendo apenas o essencial"""
    if len(messages) <= 3:  # 🎯 Máximo 3 mensagens
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

def execute_tools(state: AgentState):
    """Nó 2: Executa tools e GERA RESPOSTA NATURAL"""
    start_time = monitor.start_timer()
    
    print("="*40)
    print("🛠️ NÓ: execute_tools (Tool Execution + Resposta Natural)")
    
    tool_messages = []
    ai_message = state["messages"][-1]
    
    # 🎯 PRESERVAR O THINKING DA MENSAGEM ORIGINAL
    original_thinking = ai_message.content if ai_message.content else ""
    
    if not ai_message.tool_calls:
        print("[LOG ERRO] Última mensagem não continha Tool Calls.")
        duration = time.time() - start_time
        monitor.log_node_execution("execute_tools", duration, success=False)
        return {"messages": [AIMessage(content="Erro: Tool call não encontrado.")]}

    success = True
    tool_results = []
    
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"[LOG] Executando ferramenta: {tool_name}")
        print(f"[LOG] Argumentos: {tool_args}")
            
        if tool_name in tool_map:
            tool_func = tool_map[tool_name]
            try:
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
    if tool_results and success:
        resposta_natural = generate_natural_response(tool_results, state)
        
        # 🎯 COMBINAR THINKING ORIGINAL COM RESPOSTA NATURAL
        final_response = ""
        if original_thinking and "🧠" in original_thinking:
            final_response = f"{original_thinking}\n\n---\n\n{resposta_natural}"
        else:
            final_response = resposta_natural
            
        print(f"🎯 RESPOSTA FINAL COM THINKING: {final_response[:150]}...")
        return {"messages": [AIMessage(content=final_response)]}
    elif not success:
        error_msg = "Ocorreu um erro ao processar sua solicitação."
        if original_thinking and "🧠" in original_thinking:
            error_msg = f"{original_thinking}\n\n---\n\n{error_msg}"
        return {"messages": [AIMessage(content=error_msg)]}
    else:
        no_results_msg = "Executei as ferramentas, mas não obtive resultados."
        if original_thinking and "🧠" in original_thinking:
            no_results_msg = f"{original_thinking}\n\n---\n\n{no_results_msg}"
        return {"messages": [AIMessage(content=no_results_msg)]}

def generate_natural_response(tool_results: List[Dict], state: AgentState) -> str:
    """Gera resposta natural baseada nos resultados das tools - VERSÃO CORRIGIDA"""
    
    user_query = ""
    for msg in state["messages"]:
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
        
        elif tool_name == "schedule_appointment":
            try:
                agendamento_data = json.loads(tool_output)
                if agendamento_data.get("status") == "agendado_sucesso":
                    return f"✅ **Agendamento Confirmado!**\n\n• Médico: {agendamento_data.get('medico', 'N/A')}\n• Especialidade: {agendamento_data.get('especialidade', 'N/A')}\n• Data/Hora: {agendamento_data.get('data', 'N/A')}\n\n{agendamento_data.get('mensagem', 'Agendamento realizado com sucesso!')}"
                else:
                    return f"❌ **Não foi possível agendar:** {agendamento_data.get('mensagem', 'Erro no agendamento')}"
            except:
                return f"📅 **Agendamento Processado:**\n\n{tool_output}"
                
    except json.JSONDecodeError:
        # Se não for JSON, retornar direto
        if "erro" in tool_output.lower():
            return f"❌ **Erro no Sistema:**\n\n{tool_output}"
        else:
            return f"✅ **Operação Concluída:**\n\n{tool_output}"
    
    return f"✅ **Processamento Concluído:**\n\nSua solicitação foi processada com sucesso."

def route_to_next_step(state: AgentState) -> str:
    """Nó de Roteamento/Decisão"""
    start_time = monitor.start_timer()
    
    print("="*40)
    print("➡️ NÓ: route_to_next_step (Router)")
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        decision = "call_tool"
        print("[LOG ROTEAMENTO] Decisão: Tool Call presente. Indo para 'call_tool'.")
    elif isinstance(last_message, ToolMessage):
        decision = "call_model"
        print("[LOG ROTEAMENTO] Decisão: Tool Message recebida. Voltando para LLM.")
    else:
        decision = "end"
        print("[LOG ROTEAMENTO] Decisão: Resposta final. Terminando.")
    
    duration = time.time() - start_time
    monitor.log_node_execution("route_to_next_step", duration)
    
    return decision

# ==============================================================================
# 7. CONSTRUÇÃO DO GRAFO COM MONITOR NODE
# ==============================================================================

graph_builder = StateGraph(AgentState)

# Adicionar nós (incluindo o novo monitor node)
graph_builder.add_node("monitor", monitor_node)
graph_builder.add_node("call_model", call_model_with_tools)
graph_builder.add_node("call_tool", execute_tools)

# 🎯 APENAS UM set_entry_point - escolha UM:
graph_builder.set_entry_point("call_model")  # ⬅️ REMOVA o outro set_entry_point

# Conexões com monitoramento
graph_builder.add_edge("monitor", "call_model")

# 🆕 FLUXO LINEAR SEM VOLTAS:
graph_builder.add_conditional_edges(
    "call_model", 
    route_to_next_step, 
    {"call_tool": "call_tool", "end": END}  # 🆕 Não volta para call_model
)

graph_builder.add_edge("call_tool", END)  # 🆕 Tool sempre TERMINA - REMOVA o parêntese extra

app = graph_builder.compile()

# ==============================================================================
# 8. FUNÇÃO DE EXECUÇÃO COM RELATÓRIO FINAL
# ==============================================================================

def run_langgraph_flow(user_prompt: str):
    """Executa o LangGraph com monitoramento completo"""
    print("\n" + "#"*80)
    print(f"🎯 INÍCIO DO TESTE | USUÁRIO: {user_prompt}")
    print("#"*80)
    
    monitor.metrics["total_requests"] += 1
    initial_state = AgentState(messages=[HumanMessage(content=user_prompt)], monitoring_data={})
    
    print("\n----- FLUXO LANGGRAPH COM MONITORAMENTO -----")
    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        state_update = step[node_name]
        
        print(f"\n[FLUXO] Estado atualizado por: {node_name}")
        
        if 'messages' in state_update and state_update['messages']:
            last_message = state_update['messages'][-1]
            msg_type = type(last_message).__name__
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                msg_content = f"Tool Calls: {[tc['name'] for tc in last_message.tool_calls]}"
            elif isinstance(last_message, ToolMessage):
                msg_content = f"Tool Result: {last_message.content[:80]}..."
            else:
                msg_content = last_message.content if last_message.content else "Conteúdo vazio"
                
            print(f"[FLUXO] {msg_type}: {msg_content}")
             
    final_state = app.invoke(initial_state)
    final_answer = final_state["messages"][-1].content
    
    # RELATÓRIO FINAL DE MONITORAMENTO
    print("\n" + "🎯"*40)
    print("📊 RELATÓRIO FINAL DE MONITORAMENTO")
    print("🎯"*40)
    
    metrics = monitor.get_metrics_summary()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "*"*80)
    print(f"✅ RESPOSTA FINAL DO AGENTE:")
    print(final_answer)
    print("*"*80)

# ==============================================================================
# 9. EXECUÇÃO DOS TESTES
# ==============================================================================

if __name__ == "__main__":
    # Testes com monitoramento completo
    run_langgraph_flow("Quais especialidades temos no hospital?")
    run_langgraph_flow("Liste todos os médicos com suas especialidades")
    run_langgraph_flow("Gostaria de agendar uma consulta com cardiologista para amanhã às 10:00")