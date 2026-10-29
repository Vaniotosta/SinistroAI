
from pathlib import Path
from collections import Counter
import math
import re
import unicodedata

import streamlit as st

# =========================================================
# CONFIGURACOES
# =========================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

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

PALAVRAS_VAZIAS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "das", "do", "dos", "e", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "sem", "que", "como",
    "quando", "qual", "quais", "ao", "aos", "se", "ser", "foi",
    "sao", "sua", "suas", "seu", "seus", "mais", "muito",
    "sobre", "isso", "essa", "esse", "este", "esta"
}


def tokenizar(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )
    return [
        palavra
        for palavra in re.findall(r"[a-z0-9]+", texto)
        if len(palavra) > 1 and palavra not in PALAVRAS_VAZIAS
    ]


class ConsultaCodificada:
    def __init__(self, textos):
        self.textos = textos

    def tolist(self):
        return self.textos


class ModeloBusca:
    def encode(self, textos, **kwargs):
        # Compatibilidade com a interface de consulta do app original.
        return ConsultaCodificada(textos)


class ColecaoBusca:
    def __init__(self, trechos):
        self.trechos = trechos
        self.tokens_documentos = [
            tokenizar(item["document"]) for item in trechos
        ]
        self.frequencia_documental = Counter()

        for tokens in self.tokens_documentos:
            self.frequencia_documental.update(set(tokens))

        comprimentos = [len(tokens) for tokens in self.tokens_documentos]
        self.comprimento_medio = (
            sum(comprimentos) / len(comprimentos)
            if comprimentos else 1
        )

    def count(self):
        return len(self.trechos)

    def get(self, include=None):
        return {
            "metadatas": [
                item["metadata"] for item in self.trechos
            ]
        }

    def query(self, query_embeddings, n_results=3):
        pergunta = (
            query_embeddings[0]
            if query_embeddings else ""
        )
        termos = set(tokenizar(str(pergunta)))

        if not termos:
            return {"documents": [[]], "metadatas": [[]]}

        total = max(len(self.trechos), 1)
        resultados = []

        for indice, item in enumerate(self.trechos):
            contagens = Counter(self.tokens_documentos[indice])
            tamanho = max(len(self.tokens_documentos[indice]), 1)
            pontuacao = 0.0

            for termo in termos:
                frequencia = contagens.get(termo, 0)
                if frequencia == 0:
                    continue

                documentos_com_termo = self.frequencia_documental[termo]
                idf = math.log(
                    1 + (
                        total - documentos_com_termo + 0.5
                    ) / (
                        documentos_com_termo + 0.5
                    )
                )

                k1 = 1.5
                b = 0.75
                normalizador = frequencia + k1 * (
                    1 - b + b * tamanho / max(self.comprimento_medio, 1)
                )
                pontuacao += idf * (
                    frequencia * (k1 + 1) / normalizador
                )

            resultados.append((pontuacao, indice))

        resultados.sort(key=lambda item: item[0], reverse=True)
        resultados = [
            item for item in resultados if item[0] > 0
        ][:n_results]

        return {
            "documents": [[self.trechos[i]["document"] for _, i in resultados]],
            "metadatas": [[self.trechos[i]["metadata"] for _, i in resultados]]
        }


@st.cache_resource(show_spinner=False)
def carregar_recursos():
    pasta_documentos = PASTA_PROJETO / "documentos"

    if not pasta_documentos.exists():
        raise FileNotFoundError(
            f"Pasta de documentos não encontrada: {pasta_documentos}"
        )

    trechos = []

    for documento in sorted(pasta_documentos.glob("*.txt")):
        conteudo = documento.read_text(
            encoding="utf-8",
            errors="replace"
        )
        palavras = conteudo.split()
        numero_trecho = 0

        for inicio in range(0, len(palavras), 150):
            trecho = " ".join(palavras[inicio:inicio + 180]).strip()

            if trecho:
                trechos.append({
                    "id": f"{documento.stem}_{numero_trecho}",
                    "document": trecho,
                    "metadata": {"arquivo": documento.name}
                })
                numero_trecho += 1

    if not trechos:
        raise ValueError(
            "Nenhum documento TXT com conteúdo foi encontrado."
        )

    return ModeloBusca(), ColecaoBusca(trechos)


def consultar_ia(pergunta, contexto):
    termos = set(tokenizar(pergunta))

    padrao = re.compile(
        r"Documento:\s*(.*?)\nConteúdo:\s*(.*?)(?=\n\nDocumento:|\Z)",
        re.DOTALL
    )

    candidatos = []

    for fonte, conteudo in padrao.findall(contexto):
        sentencas = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", conteudo)
            if len(s.strip()) > 25
        ]

        if not sentencas and conteudo.strip():
            sentencas = [conteudo.strip()]

        for sentenca in sentencas:
            palavras = set(tokenizar(sentenca))
            pontos = len(termos.intersection(palavras))

            if pontos > 0:
                candidatos.append((pontos, fonte, sentenca))

    candidatos.sort(key=lambda item: item[0], reverse=True)

    selecionados = []
    vistos = set()

    for pontos, fonte, sentenca in candidatos:
        chave = (fonte, sentenca)

        if chave not in vistos:
            selecionados.append((fonte, sentenca))
            vistos.add(chave)

        if len(selecionados) >= 4:
            break

    if not selecionados:
        return (
            "Não encontrei correspondência textual clara nos trechos "
            "recuperados. Tente reformular a pergunta usando termos "
            "presentes nos documentos."
        )

    linhas = ["Trechos mais relacionados à sua pergunta:"]

    for fonte, sentenca in selecionados:
        linhas.append(f"- {sentenca}\n  Fonte: {fonte}")

    linhas.append(
        "Observação: este resultado é produzido por busca textual. "
        "Não representa uma conclusão automatizada de fraude nem "
        "uma resposta gerada por um modelo de linguagem."
    )

    return "\n\n".join(linhas)

# =========================================================
# CABECALHO
# =========================================================

st.markdown("""
<div class="brand">
    <div class="brand-label">Inteligência Analítica</div>
    <h1>SinistroAI</h1>
    <p>
        Pesquisa documental para apoio à análise de sinistros
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
        "- Pesquisa documental\n"
        "- Busca por palavras\n"
        "- Busca por termos"
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

try:
    with st.spinner("Preparando os documentos para pesquisa..."):
        modelo, colecao = carregar_recursos()

except Exception as erro:
    st.error("Não foi possível preparar a base documental.")
    with st.expander("Detalhes técnicos"):
        st.code(str(erro))
    st.stop()

if colecao.count() == 0:
    st.warning("Não existem trechos para pesquisar.")
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
        "Tipo de consulta",
        "Busca por termos"
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
        "Pesquisar documentos",
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
                "Pesquisando os trechos relevantes..."
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

    st.markdown("**Sobre esta demonstração**")
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
    Python e pesquisa documental por relevância
</div>
""", unsafe_allow_html=True)
