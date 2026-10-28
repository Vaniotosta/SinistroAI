
from pathlib import Path

import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIGURACOES
# =========================================================

PASTA_PROJETO = Path(__file__).parent
PASTA_BANCO = PASTA_PROJETO / "banco_vetorial"

MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MODELO_IA = "qwen2.5:3b"
URL_OLLAMA = "http://localhost:11434/api/chat"
NOME_COLECAO = "criterios_fraude"

st.set_page_config(
    page_title="SinistroAI | Inteligência Analítica",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# IDENTIDADE VISUAL
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #10243A;
    }

    [data-testid="stSidebar"] * {
        color: #E8EFF7;
    }

    .brand {
        color: #FFFFFF;
        background: linear-gradient(120deg, #10243A, #173D55);
        padding: 26px 30px;
        border-radius: 16px;
        border-left: 5px solid #21B6A8;
        margin-bottom: 24px;
    }

    .brand-label {
        color: #70D9CF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .brand h1 {
        color: #FFFFFF;
        font-size: 30px;
        margin: 8px 0;
        font-weight: 700;
    }

    .brand p {
        color: #D4E0EA;
        font-size: 15px;
        margin: 0;
    }

    .section-title {
        color: #10243A;
        font-size: 21px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #607287;
        font-size: 14px;
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E8F0;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E8F0;
        border-radius: 14px;
        padding: 20px;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        background-color: #087F8C;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #066873;
        color: white;
        border: none;
    }

    .answer-card {
        background-color: #FFFFFF;
        border: 1px solid #DCE6EE;
        border-left: 4px solid #087F8C;
        padding: 22px;
        border-radius: 12px;
        color: #24374B;
        line-height: 1.75;
        margin-top: 10px;
    }

    .footer {
        color: #78899B;
        font-size: 12px;
        text-align: center;
        padding: 20px 0 8px 0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# CARREGAMENTO DOS RECURSOS
# =========================================================

@st.cache_resource(show_spinner=False)
def carregar_recursos():
    modelo = SentenceTransformer(MODELO_EMBEDDING)

    cliente = chromadb.PersistentClient(
        path=str(PASTA_BANCO)
    )

    colecao = cliente.get_collection(
        name=NOME_COLECAO
    )

    return modelo, colecao


def consultar_ia(pergunta, contexto):
    prompt = f"""
Você é um assistente de inteligência analítica especializado
em apoiar a consulta a documentos de análise de sinistros.

Responda em português, de maneira clara, profissional e objetiva.
Utilize prioritariamente as informações do contexto fornecido.
Não invente fatos, regras, evidências ou prazos.
Se a informação não estiver disponível no contexto, informe
que não há informação suficiente nos documentos recuperados.
Diferencie sinais de alerta de comprovação de fraude.
Não conclua que uma pessoa cometeu fraude apenas por haver
divergências ou ocorrências repetidas.

DOCUMENTOS RECUPERADOS:
{contexto}

PERGUNTA:
{pergunta}
"""

    resposta = requests.post(
        URL_OLLAMA,
        json={
            "model": MODELO_IA,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=180
    )

    resposta.raise_for_status()

    return resposta.json()["message"]["content"]


# =========================================================
# CABECALHO
# =========================================================

st.markdown("""
<div class="brand">
    <div class="brand-label">Inteligência Analítica</div>
    <h1>SinistroAI</h1>
    <p>
        Assistente inteligente para pesquisa documental
        e análise de sinistros.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAINEL LATERAL
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ SinistroAI")
    st.caption("PLATAFORMA DE CONSULTA DOCUMENTAL")

    st.divider()

    st.markdown("### Sobre o sistema")
    st.write(
        "Pesquisa semântica e geração de respostas com "
        "base em documentos indexados."
    )

    st.markdown("### Tecnologias")
    st.markdown(
        "- Python\n"
        "- Streamlit\n"
        "- ChromaDB\n"
        "- Sentence Transformers\n"
        "- Ollama / Qwen 2.5 3B"
    )

    st.divider()

    st.markdown("### Exemplos de perguntas")
    st.caption(
        "Você pode experimentar estas perguntas no campo de consulta:"
    )

    st.markdown(
        "- Como tratar documentos divergentes?\n"
        "- O que fazer quando faltam comprovantes?\n"
        "- Como analisar sinistros recorrentes?\n"
        "- Quais verificações são recomendadas para orçamentos diferentes?"
    )

    if st.button("Limpar histórico", use_container_width=True):
        st.session_state["historico"] = []
        st.rerun()


# =========================================================
# CONFERENCIA DA BASE
# =========================================================

if not PASTA_BANCO.exists():
    st.error(
        "Banco vetorial não encontrado. "
        "Execute primeiro `python rag.py` no terminal."
    )
    st.stop()

try:
    with st.spinner("Inicializando os recursos..."):
        modelo, colecao = carregar_recursos()

except Exception as erro:
    st.error(
        "Não foi possível carregar a base de conhecimento. "
        "Confira se o banco vetorial foi criado executando `python rag.py`."
    )
    with st.expander("Detalhes técnicos"):
        st.code(str(erro))
    st.stop()

if colecao.count() == 0:
    st.warning("A base de conhecimento está vazia. Execute `python rag.py`.")
    st.stop()


# =========================================================
# INDICADORES
# =========================================================

documentos_indexados = colecao.get(
    include=["metadatas"]
)["metadatas"]

nomes_documentos = {
    item["arquivo"]
    for item in documentos_indexados
    if item and "arquivo" in item
}

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Documentos indexados",
        len(nomes_documentos)
    )

with col2:
    st.metric(
        "Trechos pesquisáveis",
        colecao.count()
    )

with col3:
    st.metric(
        "Modelo de linguagem",
        "Qwen 2.5 3B"
    )

st.write("")


# =========================================================
# FORMULARIO DE CONSULTA
# =========================================================

st.markdown(
    '<div class="section-title">Consulta inteligente</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Faça uma pergunta e consulte as informações recuperadas '
    'na base documental.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("form_consulta", clear_on_submit=True):
    pergunta = st.text_area(
        "Sua pergunta",
        placeholder=(
            "Ex.: Quais procedimentos devem ser adotados "
            "quando existem divergências nos documentos?"
        ),
        height=110
    )

    consultar = st.form_submit_button(
        "Pesquisar e gerar resposta",
        use_container_width=True
    )


# =========================================================
# PROCESSAMENTO DA CONSULTA
# =========================================================

if "historico" not in st.session_state:
    st.session_state["historico"] = []

if consultar:
    if not pergunta.strip():
        st.warning("Digite uma pergunta para continuar.")

    else:
        try:
            with st.spinner(
                "Pesquisando documentos e preparando a resposta..."
            ):
                embedding = modelo.encode(
                    [pergunta],
                    normalize_embeddings=True
                ).tolist()

                resultado = colecao.query(
                    query_embeddings=embedding,
                    n_results=min(3, colecao.count())
                )

                trechos = resultado["documents"][0]
                metadados = resultado["metadatas"][0]

                contexto = "\n\n".join(
                    f"Documento: {meta['arquivo']}\n"
                    f"Conteúdo: {trecho}"
                    for trecho, meta in zip(trechos, metadados)
                )

                resposta = consultar_ia(
                    pergunta.strip(),
                    contexto
                )

            fontes = []
            fontes_vistas = set()

            for trecho, meta in zip(trechos, metadados):
                nome = meta.get("arquivo", "Documento sem nome")

                if nome not in fontes_vistas:
                    fontes.append({
                        "arquivo": nome,
                        "trecho": trecho
                    })
                    fontes_vistas.add(nome)

            st.session_state["historico"].insert(
                0,
                {
                    "pergunta": pergunta.strip(),
                    "resposta": resposta,
                    "fontes": fontes
                }
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Não foi possível conectar ao Ollama. "
                "Verifique se o Ollama está em execução."
            )

        except requests.exceptions.Timeout:
            st.error(
                "A consulta demorou mais que o esperado. "
                "Tente novamente."
            )

        except requests.exceptions.RequestException as erro:
            st.error(f"Falha na comunicação com o Ollama: {erro}")

        except Exception as erro:
            st.error("Não foi possível concluir a consulta.")
            with st.expander("Detalhes técnicos"):
                st.code(str(erro))


# =========================================================
# RESPOSTAS E FONTES
# =========================================================

historico = st.session_state["historico"]

if historico:
    st.divider()

    st.markdown(
        '<div class="section-title">Resultado da consulta</div>',
        unsafe_allow_html=True
    )

    item_atual = historico[0]

    st.markdown("**Pergunta**")
    st.write(item_atual["pergunta"])

    st.markdown("**Resposta do assistente**")
    st.markdown(
        f'<div class="answer-card">'
        f'{item_atual["resposta"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("")
    st.markdown("**Fontes documentais recuperadas**")

    for fonte in item_atual["fontes"]:
        with st.expander(f"📄 {fonte['arquivo']}"):
            st.write(fonte["trecho"])

    if len(historico) > 1:
        st.divider()
        st.markdown(
            '<div class="section-title">Consultas anteriores</div>',
            unsafe_allow_html=True
        )

        for indice, item in enumerate(historico[1:], start=1):
            with st.expander(
                f"{indice}. {item['pergunta']}"
            ):
                st.markdown("**Resposta**")
                st.write(item["resposta"])

                st.markdown("**Fontes**")
                for fonte in item["fontes"]:
                    st.caption(f"📄 {fonte['arquivo']}")


# =========================================================
# RODAPE
# =========================================================

st.markdown("""
<div class="footer">
    SinistroAI · Pesquisa documental e IA generativa ·
    Python, ChromaDB, Sentence Transformers e Ollama
</div>
""", unsafe_allow_html=True)
