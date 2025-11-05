-- init_files/init.sql

-- ==========================================================
-- DDL (Data Definition Language) - Criação das Tabelas
-- Usando SERIAL para auto-incremento no PostgreSQL
-- ==========================================================

-- 1. TABELA DE PACIENTES
CREATE TABLE PACIENTES (
    paciente_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    data_nascimento DATE NOT NULL,
    cpf VARCHAR(14) UNIQUE,
    telefone VARCHAR(15),
    endereco VARCHAR(255)
);

-- 2. TABELA DE ESPECIALIDADES
CREATE TABLE ESPECIALIDADES (
    especialidade_id SERIAL PRIMARY KEY,
    nome_especialidade VARCHAR(100) UNIQUE NOT NULL
);

-- 3. TABELA DE MÉDICOS
CREATE TABLE MEDICOS (
    medico_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    crm VARCHAR(20) UNIQUE NOT NULL,
    especialidade_id INT,
    telefone VARCHAR(15),
    FOREIGN KEY (especialidade_id) REFERENCES ESPECIALIDADES(especialidade_id)
);

-- 4. TABELA DE CONSULTAS
CREATE TYPE TIPO_ATENDIMENTO AS ENUM('Consulta Agendada', 'Emergência', 'Retorno');

CREATE TABLE CONSULTAS (
    consulta_id SERIAL PRIMARY KEY,
    paciente_id INT NOT NULL,
    medico_id INT NOT NULL,
    data_hora TIMESTAMP WITHOUT TIME ZONE NOT NULL, -- Usando TIMESTAMP para PostgreSQL
    tipo_atendimento TIPO_ATENDIMENTO NOT NULL,    -- Usando o ENUM customizado
    valor DECIMAL(10, 2),
    FOREIGN KEY (paciente_id) REFERENCES PACIENTES(paciente_id),
    FOREIGN KEY (medico_id) REFERENCES MEDICOS(medico_id)
);

-- 5. TABELA DE PRONTUÁRIOS (Registros Clínicos Detalhados)
CREATE TABLE PRONTUARIOS (
    prontuario_id SERIAL PRIMARY KEY,
    consulta_id INT NOT NULL UNIQUE, -- Cada prontuário é de uma consulta
    diagnostico VARCHAR(255),
    historico_doenca TEXT,
    medicamentos_prescritos TEXT,
    observacoes TEXT,
    data_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (consulta_id) REFERENCES CONSULTAS(consulta_id)
);

-- ==========================================================
-- DML (Data Manipulation Language) - Inserção de Dados Fictícios
-- ==========================================================

-- Inserção de ESPECIALIDADES
INSERT INTO ESPECIALIDADES (nome_especialidade) VALUES
('Cardiologia'),
('Pediatria'),
('Dermatologia'),
('Clínica Geral'),
('Ortopedia');

-- Inserção de PACIENTES
INSERT INTO PACIENTES (nome, data_nascimento, cpf, telefone, endereco) VALUES
('Ana Clara Silva', '1995-03-10', '111.222.333-44', '(11) 98765-4321', 'Rua das Flores, 100, São Paulo'),
('Bruno Felipe Santos', '1988-11-25', '222.333.444-55', '(21) 99887-7665', 'Av. Atlântica, 500, Rio de Janeiro'),
('Carla Dias Oliveira', '2015-07-01', '333.444.555-66', '(31) 97654-3210', 'Rua da Alegria, 50, Belo Horizonte'),
('Daniela Rocha Lima', '1970-01-15', '444.555.666-77', '(41) 98888-1234', 'Travessa Paraná, 30, Curitiba');

-- Inserção de MÉDICOS (usando os IDs de ESPECIALIDADES)
INSERT INTO MEDICOS (nome, crm, especialidade_id, telefone) VALUES
('Dr. João Almeida', 'CRM/SP 123456', 1, '(11) 5555-1111'),
('Dra. Maria Barbosa', 'CRM/RJ 987654', 2, '(21) 5555-2222'),
('Dr. Pedro Costa', 'CRM/MG 456789', 4, '(31) 5555-3333');
('Dra. Ana Silva', 'CRM/SP 321098', 3, '(11) 5555-4444'), 
('Dr. Lucas Ferreira', 'CRM/BA 789012', 5, '(71) 5555-5555'),
('Dra. Laura Santos', 'CRM/RS 246801', 1, '(51) 5555-6666'), 
('Dr. Marcos Oliveira', 'CRM/PE 135792', 2, '(81) 5555-7777');

-- Inserção de CONSULTAS (usando os IDs de PACIENTES e MÉDICOS)
-- Nota: Os IDs de 1 a 4 são atribuídos automaticamente
INSERT INTO CONSULTAS (paciente_id, medico_id, data_hora, tipo_atendimento, valor) VALUES
(1, 1, '2025-10-20 10:00:00', 'Consulta Agendada', 250.00),
(3, 2, '2025-10-21 14:30:00', 'Retorno', 180.00),
(4, 3, '2025-10-22 09:15:00', 'Emergência', 300.00),
(1, 3, '2025-10-23 11:00:00', 'Consulta Agendada', 200.00);

-- Inserção de PRONTUÁRIOS (usando os IDs de CONSULTAS)
INSERT INTO PRONTUARIOS (consulta_id, diagnostico, historico_doenca, medicamentos_prescritos, observacoes) VALUES
(1, 
 'Hipertensão Essencial Controlada', 
 'Paciente com histórico familiar de hipertensão. Mantém pressão arterial controlada com medicação contínua.', 
 'Losartana 50mg, AAS 100mg', 
 'Solicitado eletrocardiograma de rotina. Próximo retorno em 6 meses.'),
(2, 
 'Resfriado Comum', 
 'Criança apresenta coriza e tosse há 3 dias. Sem febre. Histórico vacinal em dia.', 
 'Paracetamol (em caso de dor), Lavagem nasal com soro fisiológico', 
 'Orientação aos pais sobre hidratação. Manter observação.'),
(3, 
 'Crise de Enxaqueca', 
 'Paciente chegou com dor de cabeça intensa, náuseas e fotofobia. Histórico de enxaqueca crônica.', 
 'Dipirona injetável (aplicado no consultório), Sumatriptano (para casa)', 
 'Orientado a evitar gatilhos (cafeína, estresse). Necessário agendar consulta de rotina.'),
(4,
 'Acompanhamento de rotina (Check-up)',
 'Paciente refere cansaço leve, mas nega dor ou sintoma específico. Exames laboratoriais normais.',
 'Multivitamínico (suplemento)',
 'Encaminhada para nutricionista. Retorno necessário em 1 mês para acompanhamento da pressão arterial.');