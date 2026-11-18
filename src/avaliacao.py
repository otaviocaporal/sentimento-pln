import os
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score

from src.modelos import criar_modelo


def avaliar_modelos(dados):

    # separar textos e sentimentos
    X = dados["texto"]
    y = dados["sentimento"]

    # Divisão: 75% treinamento e 25% teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # validacao cruzada estratificada
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # metricas
    metricas = {
        "acc": "accuracy",
        "f1": make_scorer(f1_score, pos_label="positivo")
    }

    # quatro configs
    configuracoes = {
        "A": (False, "bow", "mantidas", "BoW"),
        "B": (False, "tfidf", "mantidas", "TF-IDF"),
        "C": (True, "bow", "removidas", "BoW"),
        "D": (True, "tfidf", "removidas", "TF-IDF")
    }

    resultados = []

    for nome, parametros in configuracoes.items():

        remover_stopwords, tipo, stopwords, vetorizador = parametros

        modelo = criar_modelo(
            remover_stopwords=remover_stopwords,
            tipo_vetorizador=tipo
        )

        # validacao cruzada somente no treino
        scores = cross_validate(
            modelo,
            X_treino,
            y_treino,
            cv=cv,
            scoring=metricas,
            error_score="raise"
        )

        # ajustar modelo no treino para descobrir o vocabulario
        modelo.fit(X_treino, y_treino)

        vocabulario = len(
            modelo.named_steps["vetorizador"].vocabulary_
        )

        resultados.append({
            "config": nome,
            "stopwords": stopwords,
            "vetorizador": vetorizador,
            "vocabulario": vocabulario,
            "acc_cv_media": scores["test_acc"].mean(),
            "acc_cv_desvio": scores["test_acc"].std(),
            "f1_cv_media": scores["test_f1"].mean(),
            "f1_cv_desvio": scores["test_f1"].std()
        })

        print(f"\nConfiguração {nome}")
        print(f"Acurácia CV: {scores['test_acc'].mean():.4f}")
        print(f"F1 CV: {scores['test_f1'].mean():.4f}")
        print(f"Vocabulário: {vocabulario}")

    # tabela de comparacao
    tabela = pd.DataFrame(resultados)

    # salvar comparacao
    os.makedirs("resultados", exist_ok=True)
    tabela.to_csv(
        "resultados/comparacao.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # escolher vencedor pela media do F1 - validacao cruzada
    melhor = tabela.sort_values(
        "f1_cv_media",
        ascending=False
    ).iloc[0]

    print("\nMelhor configuração:")
    print(melhor)

    print("\nTabela completa:")
    print(tabela)

    return tabela, X_treino, X_teste, y_treino, y_teste, melhor


from sklearn.metrics import confusion_matrix, classification_report
from src.modelos import criar_modelo


def avaliar_modelo_final(X_treino, X_teste, y_treino, y_teste, melhor):

    configuracoes = {
        "A": (False, "bow"),
        "B": (False, "tfidf"),
        "C": (True, "bow"),
        "D": (True, "tfidf")
    }

    remover_stopwords, tipo = configuracoes[melhor["config"]]

    modelo = criar_modelo(remover_stopwords, tipo)

    # treinar usando todo o conjunto de treinamento
    modelo.fit(X_treino, y_treino)

    # previsoes no teste
    previsoes = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)

    indice_positivo = list(modelo.classes_).index("positivo")
    prob_positivo = probabilidades[:, indice_positivo]

    os.makedirs("resultados", exist_ok=True)

    # matriz de confusao
    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=["negativo", "positivo"]
    )

    pd.DataFrame(
        matriz,
        index=["real_negativo", "real_positivo"],
        columns=["previsto_negativo", "previsto_positivo"]
    ).to_csv("resultados/matriz_confusao.csv")

    # relatorio de classificacao
    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=["negativo", "positivo"],
        zero_division=0
    )

    with open("resultados/relatorio_classificacao.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    # registrar erros
    erros = pd.DataFrame({
        "texto": X_teste.values,
        "real": y_teste.values,
        "previsto": previsoes,
        "probabilidade": prob_positivo
    })

    erros = erros[erros["real"] != erros["previsto"]]
    erros.to_csv("resultados/erros.csv", index=False, encoding="utf-8-sig")

    # palavras influentes
    vetorizador = modelo.named_steps["vetorizador"]
    classificador = modelo.named_steps["classificador"]

    palavras = vetorizador.get_feature_names_out()
    coeficientes = classificador.coef_[0]

    indices_positivos = coeficientes.argsort()[-10:][::-1]
    indices_negativos = coeficientes.argsort()[:10]

    influentes = []

    for indice in indices_positivos:
        influentes.append({
            "palavra": palavras[indice],
            "classe": "positivo",
            "coeficiente": coeficientes[indice]
        })

    for indice in indices_negativos:
        influentes.append({
            "palavra": palavras[indice],
            "classe": "negativo",
            "coeficiente": coeficientes[indice]
        })

    pd.DataFrame(influentes).to_csv(
        "resultados/palavras_influentes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAvaliação final:")
    print("Matriz de confusão:")
    print(matriz)
    print("\nRelatório de classificação:")
    print(relatorio)
    print(f"Quantidade de erros: {len(erros)}")

    return modelo


def avaliar_casos_dificeis(modelo):

    casos = pd.read_csv("data/casos_dificeis.csv")

    previsoes = modelo.predict(casos["texto"])

    probabilidades = modelo.predict_proba(casos["texto"])
    indice_positivo = list(modelo.classes_).index("positivo")

    resultados = pd.DataFrame({
        "texto": casos["texto"],
        "tipo": casos["tipo"],
        "esperado": casos["sentimento_esperado"],
        "previsto": previsoes,
        "probabilidade_positivo": probabilidades[:, indice_positivo]
    })

    resultados.to_csv(
        "resultados/casos_dificeis_predicoes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCasos difíceis:")
    print(resultados[["tipo", "esperado", "previsto"]])

