# 🛡️ SinistroAI — Assistente Inteligente para Análise de Sinistros
Link APP: https://sinistroai-5yqt2oc4w7qrpjv9hxpzuk.streamlit.app/

## 📌 Visão geral

O **SinistroAI** é um projeto de Inteligência Artificial Generativa que utiliza a arquitetura **Retrieval-Augmented Generation (RAG)** para consultar documentos relacionados à análise de sinistros de seguros.

A aplicação combina busca semântica e um modelo de linguagem local para recuperar informações relevantes e gerar respostas contextualizadas.

O objetivo é demonstrar como técnicas de IA podem apoiar a pesquisa documental, a organização de informações e a análise de situações que merecem investigação adicional.

## 🎯 Objetivos do projeto

- Implementar uma arquitetura RAG de ponta a ponta.
- Criar uma base vetorial para recuperação semântica de documentos.
- Integrar um modelo de linguagem para geração de respostas.
- Desenvolver uma interface interativa com Streamlit.
- Exibir os documentos recuperados como fontes de consulta.
- Orientar respostas que diferenciem sinais de alerta de comprovação de fraude.

## 🧠 Tecnologias utilizadas

- **Python** — desenvolvimento da aplicação.
- **Streamlit** — interface web interativa.
- **ChromaDB** — armazenamento e recuperação vetorial.
- **Sentence Transformers** — geração de embeddings multilíngues.
- **Ollama** — execução local do modelo de linguagem.
- **Qwen 2.5 3B** — modelo utilizado na geração das respostas.

## ⚙️ Arquitetura

1. Os documentos de referência são carregados.
2. O conteúdo é dividido em trechos menores.
3. Os trechos são convertidos em embeddings.
4. Os embeddings são armazenados no ChromaDB.
5. A pergunta do usuário é convertida em embedding.
6. O sistema recupera os trechos mais relevantes.
7. O modelo de linguagem gera uma resposta baseada no contexto recuperado.
8. A interface apresenta a resposta e as fontes documentais.

## 📂 Estrutura do projeto

```text
projeto_rag_fraudes/
├── app.py
├── rag.py
├── rag_chat.py
├── documentos/
│   ├── criterios_fraude.txt
│   ├── procedimentos_sinistros.txt
│   ├── sinais_alerta.txt
│   └── estudos_de_caso.txt
├── banco_vetorial/
└── README.md
```

## 🚀 Execução local

### 1. Pré-requisitos

- Python 3.11
- Ollama instalado
- Git (opcional para execução local)

### 2. Instale as dependências

```bash
pip install streamlit chromadb sentence-transformers requests
```

### 3. Baixe o modelo de linguagem

```bash
ollama pull qwen2.5:3b
```

### 4. Prepare a base vetorial

```bash
python rag.py
```

### 5. Inicie a aplicação

```bash
streamlit run app.py
```

A aplicação será disponibilizada localmente, normalmente em `http://localhost:8501`.

## 🔎 Exemplos de consultas

- Como tratar divergências entre documentos?
- Quais verificações são recomendadas quando faltam comprovantes?
- Como analisar a ocorrência de sinistros recorrentes?
- Quais explicações alternativas devem ser consideradas durante a análise?

## ⚠️ Limitações

Este projeto tem finalidade educacional e utiliza documentos fictícios. As respostas geradas dependem da qualidade dos documentos recuperados e do modelo de linguagem utilizado.

Sinais de alerta não comprovam fraude. As respostas devem ser verificadas e não substituem uma investigação profissional, uma análise jurídica ou uma decisão formal sobre um sinistro.

## 👨‍💻 Autor

**Vânio Tosta**

Projeto de portfólio voltado à aplicação de Ciência de Dados, busca semântica e Inteligência Artificial Generativa em um cenário de análise de sinistros.

---

*Desenvolvido para fins de aprendizado e demonstração técnica.*
