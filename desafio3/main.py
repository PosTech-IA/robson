# main.py

import sys
import os
import operator
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage
import logging
# Importar o grafo e o monitor
from agent_graph import build_agent_graph, monitor
from config import DB_NAME

# ==============================================================================
# 8. FUNÇÕES DE EXECUÇÃO
# ==============================================================================

def invoke_agent(graph_app, user_input: str):
    """
    Invoca o agente com uma query de usuário.
    """
    print("\n" + "#"*70)
    print(f"🤖 AGENTE Dr. IA INICIADO | Query: {user_input}")
    print("#"*70)
    
    # Inicia o estado com a mensagem do usuário
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "monitoring_data": {}
    }
    
    monitor.metrics["total_requests"] += 1
    
    final_output = None
    
    try:
        # Invocar o grafo
        for step in graph_app.stream(initial_state):
            name, state = next(iter(step.items()))
            if name == 'monitor_end':
                final_output = state
                break

        # Processar a saída final
        if final_output and final_output.get('messages'):
            last_message = final_output['messages'][-1]
            print("\n" + "="*70)
            print("✨ RESPOSTA FINAL DO AGENTE ✨")
            print("="*70)
            print(last_message.content)
            print("="*70)
        else:
            print("\n⚠️ O grafo não retornou uma mensagem final.")
            
    except Exception as e:
        monitor.metrics["errors"] += 1
        print("\n" + "🚨"*10)
        print(f"[DrIA_Agent|ERROR]🚨 ERRO: Exceção fatal durante a execução do grafo: {e}")
        import traceback
        traceback.print_exc()
        print("🚨"*10)

def run_tests(app):
    """
    Executa um conjunto fixo de queries para teste.
    """
    print("\n" + "---"*20)
    print("🧪 MODO DE TESTE (AGENT_MODE=TEST) ATIVADO 🧪")
    print("---"*20)
    
    test_queries = [
        "Quais especialidades temos no hospital?",
        "Liste todos os médicos com suas especialidades",
        "Gostaria de agendar uma consulta com cardiologista para amanhã às 10:00"
    ]
    
    for query in test_queries:
        invoke_agent(app, query)

def run_interactive_chat(app):
    """
    Executa o loop de chat interativo.
    """
    print("\n" + "---"*20)
    print("💬 MODO INTERATIVO (CHAT) ATIVADO 💬")
    print(f"HOSPITAL: {DB_NAME}")
    print("---"*20)
    
    while True:
        try:
            user_input = input(f"\n\nPergunte ao Dr. IA ({DB_NAME}) ou 'sair':\n> ")
            if user_input.lower() == 'sair':
                print("\nPrograma encerrado.")
                break
            
            if user_input.strip():
                invoke_agent(app, user_input)
                
        except KeyboardInterrupt:
            print("\nPrograma encerrado por interrupção do usuário.")
            break
        except Exception as e:
            print(f"Erro inesperado no loop principal: {e}")

def main():
    """Função principal de execução"""
    logging.basicConfig(
        level=logging.INFO,  # Define o nível mínimo de log a ser exibido (INFO, DEBUG, ERROR)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 1. Construir e compilar o grafo
    app = build_agent_graph()

    # 2. Verificar o modo de execução via Variável de Ambiente
    agent_mode = os.getenv('AGENT_MODE', 'TESTE').upper() # Default é CHAT
    
    if agent_mode == 'TESTE':
        run_tests(app)
    else:
        run_interactive_chat(app)

    # 3. Exibir métricas finais
    print("\n" + "---"*20)
    print("📈 SUMÁRIO FINAL DE MONITORAMENTO")
    print("---"*20)
    print(json.dumps(monitor.get_metrics_summary(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()