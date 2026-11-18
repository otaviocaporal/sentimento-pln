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