# monitoring.py

import time
import json  # Import adicionado
from typing import Dict, Any
import logging
from datetime import datetime

class MonitoringSystem:
    def __init__(self, logger_name='monitoring'):
        self.metrics = {
            "total_requests": 0,
            "node_executions": {},
            "tool_calls": {},
            "errors": 0,
            "start_time": time.time()
        }
        
        # Configurar logger
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start_timer(self):
        """Inicia um timer para medir duração de execução."""
        return time.time()
    
    def log_node_execution(self, node_name: str, duration: float, success: bool = True):
        """Registra a execução de um nó do grafo."""
        if node_name not in self.metrics["node_executions"]:
            self.metrics["node_executions"][node_name] = {
                "count": 0,
                "total_duration": 0,
                "successes": 0,
                "failures": 0
            }
        
        node_stats = self.metrics["node_executions"][node_name]
        node_stats["count"] += 1
        node_stats["total_duration"] += duration
        
        if success:
            node_stats["successes"] += 1
        else:
            node_stats["failures"] += 1
        
        self.logger.info(f"Node '{node_name}' executado em {duration:.2f}s (sucesso: {success})")
    
    def log_tool_call(self, tool_name: str):
        """Registra uma chamada de ferramenta."""
        if tool_name not in self.metrics["tool_calls"]:
            self.metrics["tool_calls"][tool_name] = 0
        
        self.metrics["tool_calls"][tool_name] += 1
        self.logger.debug(f"Tool chamada: {tool_name}")
    
    def print_real_time_metrics(self, context: str, state: Dict[str, Any]):
        """Exibe métricas em tempo real durante a execução."""
        total_nodes = len(self.metrics["node_executions"])
        total_tools = sum(self.metrics["tool_calls"].values())
        
        print(f"\n**MONITORAMENTO ({context})**")
        print(f"   • Total de Requisições: {self.metrics['total_requests']}")
        print(f"   • Nós Executados: {total_nodes}")
        print(f"   • Chamadas de Tools: {total_tools}")
        print(f"   • Erros: {self.metrics['errors']}")
        
        if self.metrics["node_executions"]:
            print("   • Últimos nós executados:")
            for node, stats in list(self.metrics["node_executions"].items())[-3:]:
                avg_time = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
                print(f"     - {node}: {stats['count']}x (avg: {avg_time:.2f}s)")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retorna um resumo completo das métricas coletadas."""
        total_duration = time.time() - self.metrics["start_time"]
        
        summary = {
            "session_duration_seconds": round(total_duration, 2),
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["errors"],
            "nodes_executed": {},
            "tools_called": self.metrics["tool_calls"]
        }
        
        # Calcular estatísticas dos nós
        for node_name, stats in self.metrics["node_executions"].items():
            avg_time = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            success_rate = (stats["successes"] / stats["count"] * 100) if stats["count"] > 0 else 0
            
            summary["nodes_executed"][node_name] = {
                "execution_count": stats["count"],
                "total_time_seconds": round(stats["total_duration"], 2),
                "average_time_seconds": round(avg_time, 2),
                "successes": stats["successes"],
                "failures": stats["failures"],
                "success_rate_percent": round(success_rate, 2)
            }
        
        # Log do resumo
        self.logger.info(f"Resumo da Sessão: {json.dumps(summary, indent=4, ensure_ascii=False)}")
        
        return summary
    
    def reset_metrics(self):
        """Reseta todas as métricas para uma nova sessão."""
        self.metrics = {
            "total_requests": 0,
            "node_executions": {},
            "tool_calls": {},
            "errors": 0,
            "start_time": time.time()
        }
        self.logger.info("Métricas resetadas para nova sessão")