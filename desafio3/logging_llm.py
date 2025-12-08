import logging
import os

# Define o diretório para os logs
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger(name, log_file, level=logging.INFO):
    """
    Função para configurar um logger nomeado com handlers de arquivo e console.
    """
    # Cria um logger personalizado
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita a duplicação de logs se o logger já tiver handlers
    if logger.handlers:
        return logger

    # Handler para escrever em um arquivo (todos os níveis, incluindo DEBUG)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    file_handler.setLevel(logging.DEBUG)

    # Handler para o console (nível INFO ou superior)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter para definir o formato das mensagens de log
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Adiciona o formatter aos handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

