Análise de Sentimento com PLN

Projeto desenvolvido para a disciplina de Fundamentos de Inteligência Artificial.

Objetivo

Desenvolver um classificador de sentimentos para avaliações de clientes, identificando textos positivos e negativos utilizando técnicas de Processamento de Linguagem Natural.

Tecnologias utilizadas
Python 3.12
Pandas
NumPy
Scikit-learn
Matplotlib
Instalação

Instale as dependências com:

bash
pip install -r requirements.txt

Execução

Execute o projeto pela pasta principal:

bash
python main.py

Modelos avaliados

Foram comparadas quatro configurações:

BoW com stopwords mantidas.
TF-IDF com stopwords mantidas.
BoW com stopwords removidas.
TF-IDF com stopwords removidas.

Foi utilizada Regressão Logística e validação cruzada estratificada com cinco folds.

Os resultados são salvos automaticamente na pasta resultados/.