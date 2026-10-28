

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Configurações do projeto
PASTA_PROJETO = Path(__file__).parent
PASTA_DOCUMENTOS = PASTA_PROJETO / "documentos"
PASTA_BANCO = PASTA_PROJETO / "banco_vetorial"

MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

NOME_COLECAO = "criterios_fraude"


def dividir_texto(texto, tamanho=180, sobreposicao=30):
    """Divide o texto em trechos menores, preservando alguma sobreposição."""
    palavras = texto.split()
    trechos = []
    inicio = 0

    while inicio < len(palavras):
        fim = min(inicio + tamanho, len(palavras))
        trecho = " ".join(palavras[inicio:fim])
        trechos.append(trecho)

        if fim == len(palavras):
            break

        inicio = fim - sobreposicao

    return trechos


def main():
    arquivos = sorted(PASTA_DOCUMENTOS.glob("*.txt"))

    if not arquivos:
        print("Nenhum arquivo .txt foi encontrado na pasta documentos.")
        return

    print(f"Documentos encontrados: {len(arquivos)}")

    todos_trechos = []
    metadados = []
    ids = []

    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8")
        trechos = dividir_texto(texto)

        for numero, trecho in enumerate(trechos):
            todos_trechos.append(trecho)
            metadados.append({
                "arquivo": arquivo.name,
                "trecho": numero
            })
            ids.append(f"{arquivo.stem}_{numero}")

        print(f"- {arquivo.name}: {len(trechos)} trecho(s)")

    print("\nCarregando o modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_EMBEDDING)

    print("Gerando embeddings...")
    embeddings = modelo.encode(
        todos_trechos,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    print("\nPreparando o banco vetorial...")
    cliente = chromadb.PersistentClient(path=str(PASTA_BANCO))

    # Recria apenas a coleção utilizada por este projeto.
    try:
        cliente.delete_collection(name=NOME_COLECAO)
    except Exception:
        pass

    colecao = cliente.create_collection(name=NOME_COLECAO)

    colecao.add(
        ids=ids,
        documents=todos_trechos,
        embeddings=embeddings,
        metadatas=metadados
    )

    print("\nIndexação concluída!")
    print(f"Documentos processados: {len(arquivos)}")
    print(f"Trechos armazenados: {colecao.count()}")
    print(f"Banco salvo em: {PASTA_BANCO}")


if __name__ == "__main__":
    main()

