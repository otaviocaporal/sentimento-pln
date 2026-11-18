import re

STOPWORDS = {
    "o", "a", "os", "as",
    "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "por", "para", "com",
    "e", "ou", "mas",
    "ao", "aos", "à", "eu"
}

def preprocessar_texto(texto, remover_stopwords=False):

    texto = texto.lower()

    palavras = re.findall(
        r"[a-záàâãéêíóôõúçü]+",
        texto
    )

    if remover_stopwords:
        palavras = [
            palavra for palavra in palavras
            if palavra not in STOPWORDS
        ]

    return " ".join(palavras)

if __name__ == "__main__":
    texto = "Eu não gostei do produto!!!"

    print("Sem remover stopwords:")
    print(preprocessar_texto(texto))

    print("\nRemovendo stopwords:")
    print(preprocessar_texto(texto, remover_stopwords=True))