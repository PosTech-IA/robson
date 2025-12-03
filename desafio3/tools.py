# db_tools.py (REFATORADO)

import json
from typing import Dict, Any

import psycopg2 
from psycopg2 import extras 
from langchain_core.tools import tool

from config import DB_CONFIG, DB_NAME

# ==============================================================================
# 2. FUNÇÕES DO BANCO DE DADOS
# ==============================================================================

# Definição das tabelas válidas para consultas (SELECT)
VALID_READ_TABLES = ['PACIENTES', 'ESPECIALIDADES', 'MEDICOS', 'CONSULTAS', 'PRONTUARIOS']
# Definição das tabelas válidas para escrita (INSERT)
VALID_WRITE_TABLES = ['CONSULTAS'] 


def execute_sql_query_impl(query: str) -> str:
    """Implementação da execução SQL (apenas SELECT) com validação de segurança."""
    print("\n" + "~"*50)
    print(f"[EXECUÇÃO DA TOOL] Iniciando execução de Query SQL (READ).")
    print(f"[QUERY BRUTA] {query}")
    
    query_lower = query.lower().strip()
    
    # 🎯 Segurança: Apenas SELECT
    if not query_lower.startswith('select'):
        error_msg = "ERRO: Apenas consultas SELECT são permitidas."
        print(f"[DB ERRO] {error_msg}")
        return json.dumps({"status": "erro_seguranca", "mensagem": error_msg}, ensure_ascii=False)
    
    # 🎯 Validação de Tabelas
    table_used = None
    for table in VALID_READ_TABLES:
        if table.lower() in query_lower:
            table_used = table
            break
    
    if not table_used:
        error_msg = f"ERRO: Tabela não encontrada. Use apenas: {', '.join(VALID_READ_TABLES)}"
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
            error_suggestion = f" Use apenas as tabelas em MAIÚSCULAS: {', '.join(VALID_READ_TABLES)}."
            
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

def _execute_sql_write_impl(query: str, table_name: str) -> bool:
    """Implementação da execução SQL (INSERT/UPDATE/DELETE) com commit."""
    if table_name not in VALID_WRITE_TABLES:
        print(f"[DB ERRO WRITE] Tabela não permitida para escrita: {table_name}")
        return False
        
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
            return True
    except psycopg2.Error as e:
        print(f"[DB ERRO WRITE] Falha de EXECUÇÃO SQL de escrita: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


@tool
def SQL_query_tool(query: str) -> str:
    """Executa uma consulta SELECT SQL no banco de dados"""
    return execute_sql_query_impl(query)

# ==============================================================================
# 3. NOVAS FERRAMENTAS DE NEGÓCIO
# ==============================================================================

@tool
def check_and_schedule_availability(medico_id: int, data_hora: str, paciente_id: int) -> str:
    """
    VERIFICA se um médico está livre e AGENDA a consulta na tabela CONSULTAS.
    Esta ferramenta DEVE ser usada para finalizar um pedido de agendamento.
    Requer o ID exato do médico, a data/hora exata (YYYY-MM-DD HH:MI:SS) e o ID do paciente.
    """
    print("\n" + "~"*50)
    print(f"[EXECUÇÃO DA TOOL] Verificação e Agendamento de Consulta.")
    print(f"Dados: Médico ID={medico_id}, Data/Hora={data_hora}, Paciente ID={paciente_id}")

    # 1. Checar Disponibilidade (Query SELECT)
    availability_query = f"""
    SELECT consulta_id
    FROM CONSULTAS
    WHERE medico_id = {medico_id} AND data_hora = '{data_hora}';
    """
    
    # Reutiliza a função de leitura para checar
    try:
        result_json = execute_sql_query_impl(availability_query)
        result_data = json.loads(result_json)
    except Exception as e:
        print(f"[DB ERRO] Falha ao checar disponibilidade: {e}")
        return json.dumps({
            "status": "erro_verificacao",
            "mensagem": f"Erro interno ao checar disponibilidade: {e}"
        })

    if result_data:
        # Consulta já existe, médico ocupado
        print(f"[AGENDAMENTO] Médico {medico_id} está OCUPADO em {data_hora}.")
        return json.dumps({
            "status": "data_indisponivel",
            "mensagem": f"O médico com ID {medico_id} já possui uma consulta agendada para {data_hora}. Por favor, escolha outro horário."
        })
    else:
        # 2. Agendar (Query INSERT)
        insert_query = f"""
        INSERT INTO CONSULTAS (medico_id, paciente_id, data_hora)
        VALUES ({medico_id}, {paciente_id}, '{data_hora}');
        """
        
        # Usa a nova função de escrita
        success = _execute_sql_write_impl(insert_query, "CONSULTAS")

        if success:
            print(f"[AGENDAMENTO] ✅ Consulta agendada com sucesso!")
            # Nota: Em um sistema real, você retornaria o ID da nova consulta
            return json.dumps({
                "status": "agendado_sucesso",
                "medico_id": medico_id,
                "data_hora": data_hora,
                "paciente_id": paciente_id,
                "mensagem": f"Consulta agendada com sucesso com o médico ID {medico_id} para {data_hora}."
            })
        else:
            print("[AGENDAMENTO] ❌ Falha ao salvar a consulta no banco.")
            return json.dumps({
                "status": "erro_persistente",
                "mensagem": "Falha ao salvar a consulta no banco de dados. Tente novamente mais tarde."
            })
    print("~"*50)


# OBSERVAÇÃO: A antiga tool 'schedule_appointment' foi removida, 
# pois a nova 'check_and_schedule_availability' é mais completa e robusta.

tools = [SQL_query_tool, check_and_schedule_availability]
tool_map = {tool.name: tool for tool in tools}