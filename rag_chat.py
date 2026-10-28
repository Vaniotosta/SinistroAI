
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

PASTA_PROJETO = Path(__file__).parent
PASTA_BANCO = PASTA_PROJETO / "banco_vetorial"

MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MODELO_IA = "qwen2.5:3b"
URL_OLLAMA = "http://localhost:11434/api/chat"


def main():
    print("Carregando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDING)

    cliente = chromadb.PersistentClient(path=str(PASTA_BANCO))
    colecao = cliente.get_collection(name="criterios_fraude")

    print(f"\nBase carregada: {colecao.count()} trechos.")
    print("Faça perguntas sobre os documentos.")
    print("Digite 'sair' para encerrar.")

    while True:
        pergunta = input("\nSua pergunta: ").strip()

        if pergunta.lower() == "sair":
            print("Encerrando o assistente.")
            break

        if not pergunta:
            continue

        embedding_pergunta = modelo.encode(
            [pergunta],
            normalize_embeddings=True
        ).tolist()

        resultado = colecao.query(
            query_embeddings=embedding_pergunta,
            n_results=min(3, colecao.count())
        )

        trechos = resultado["documents"][0]
        metadados = resultado["metadatas"][0]

        contexto = "\n\n".join(
            f"Fonte: {meta['arquivo']}\nConteúdo: {trecho}"
            for trecho, meta in zip(trechos, metadados)
        )

        prompt = f"""
Você é um assistente de apoio à análise de sinistros de seguros.

Responda em português, usando as informações dos documentos fornecidos.
Não invente fatos, regras ou evidências.
Se os documentos não responderem à pergunta, informe que não há
informação suficiente na base consultada.
Diferencie indícios de prova de fraude.
Não recomende negar uma indenização apenas por existir um sinal de alerta.

DOCUMENTOS RECUPERADOS:
{contexto}

PERGUNTA:
{pergunta}
"""

        print("\nConsultando o modelo de IA...")

        try:
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
            texto_resposta = resposta.json()["message"]["content"]

            print("\nRESPOSTA DO ASSISTENTE:\n")
            print(texto_resposta)

            print("\nFONTES RECUPERADAS:")
            fontes_exibidas = set()

            for meta in metadados:
                nome = meta["arquivo"]
                if nome not in fontes_exibidas:
                    print(f"- {nome}")
                    fontes_exibidas.add(nome)

        except requests.exceptions.RequestException as erro:
            print("Erro ao consultar o Ollama:", erro)

        except (KeyError, IndexError) as erro:
            print("Erro ao interpretar a resposta:", erro)


if __name__ == "__main__":
    main()
