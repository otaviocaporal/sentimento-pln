import pandas as pd
import matplotlib.pyplot as plt

# dataset
dados = pd.read_csv("data/Avaliacoes.csv")

# primeiras avaliações
print("Primeiras avaliações:")
print(dados.head())

# informações dataset
print("\nInformações do dataset:")
dados.info()

# quantidade total de avaliações
print("\nTotal de avaliações:")
print(len(dados))

# distribuição dos sentimentos
print("\nDistribuição dos sentimentos:")
distribuicao = dados["sentimento"].value_counts()

print(distribuicao)

# percentual de cada sentimento
print("\nPorcentagem dos sentimentos:")
print(dados["sentimento"].value_counts(normalize=True) * 100)

# calcular quantidade de palavras em cada avaliação
dados["quantidade_palavras"] = dados["texto"].apply(
    lambda texto: len(texto.split())
)

# média de palavras por sentimento
print("\nMédia de palavras por sentimento:")

media_palavras = dados.groupby("sentimento")["quantidade_palavras"].mean()

print(media_palavras)

# exemplos avaliacoes positivas
print("\nExemplos de avaliações positivas:")
positivos = dados[dados["sentimento"] == "positivo"].head(3)

for _, linha in positivos.iterrows():
    print(linha["texto"])

# exemplos avaliacoes negativas
print("\nExemplos de avaliações negativas:")
negativos = dados[dados["sentimento"] == "negativo"].head(3)

for _, linha in negativos.iterrows():
    print(linha["texto"])


# gráfico 1 - distribuição dos sentimentos
plt.figure(figsize=(8, 5))
plt.bar(
    distribuicao.index,
    distribuicao.values,
    color=["red", "green"]
)
plt.title("Distribuição dos sentimentos")
plt.xlabel("Sentimento")
plt.ylabel("Quantidade de avaliações")
plt.savefig("resultados/distribuicao_sentimentos.png")
plt.show()

# gráfico 2 - média de palavras por sentimento
plt.figure(figsize=(8, 5))
plt.bar(
    media_palavras.index,
    media_palavras.values,
    color=["red", "green"]
)
plt.title("Média de palavras por sentimento")
plt.xlabel("Sentimento")
plt.ylabel("Média de palavras")
plt.savefig("resultados/media_palavras.png")
plt.show()

