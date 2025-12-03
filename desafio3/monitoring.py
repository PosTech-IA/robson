# monitoring.py

import time
import uuid
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import AIMessage, ToolMessage

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