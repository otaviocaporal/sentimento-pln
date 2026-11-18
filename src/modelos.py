from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessamento import preprocessar_texto


def criar_modelo(remover_stopwords=False, tipo_vetorizador="bow"):

    if tipo_vetorizador == "bow":
        vetorizador = CountVectorizer()

    elif tipo_vetorizador == "tfidf":
        vetorizador = TfidfVectorizer()

    else:
        raise ValueError("Vetorizador inválido")

    modelo = Pipeline([
        ("preprocessamento", FunctionTransformer(
            lambda textos: [
                preprocessar_texto(
                    texto,
                    remover_stopwords=remover_stopwords
                )
                for texto in textos
            ]
        )),
        ("vetorizador", vetorizador),
        ("classificador", LogisticRegression(max_iter=1000, random_state=42))
    ])

    return modelo

if __name__ == "__main__":
    modelo_a = criar_modelo(False, "bow")
    modelo_b = criar_modelo(False, "tfidf")
    modelo_c = criar_modelo(True, "bow")
    modelo_d = criar_modelo(True, "tfidf")

    print("Modelo A:", modelo_a)
    print("Modelo B:", modelo_b)
    print("Modelo C:", modelo_c)
    print("Modelo D:", modelo_d)
