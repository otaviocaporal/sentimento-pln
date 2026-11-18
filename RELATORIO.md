# RELATORIO — Análise de Sentimento com PLN

**Nome: Otávio Brocca Caporal** 
**Repositório: https://github.com/otaviocaporal/sentimento-pln.git** 
**Data: 17/11/2026**

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420| 0,8564 ± 0,0926| 0,8602 ± 0,0899|
| B | mantidas | TF-IDF | 420| 0,8462 ± 0,0843| 0,8440 ± 0,0875|
| C | removidas | BoW | 406| 0,8564 ± 0,0804| 0,8584 ± 0,0806|
| D | removidas | TF-IDF | 406| 0,8410 ± 0,0714| 0,8391 ± 0,0752|

**Configuração escolhida:** A - BoW com stopword mantidas **Acurácia no teste:** 0,82 (82%) **F1 no teste:** 0,81

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*

No pré-processamento, usei o .lower() para converter os textos para letras minúsculas. Também removi pontuações e números, mas manti letras com acentos. Comparei duas configurações diferentes: uma mantendo as stopwords e outra removendo elas.
O vocabulário ficou com 420 palavras mantendo as stopwords e 406 removendo elas. A palavra "não" não foi adicionada na lista pois é importante para identificar frases de negações e não alterar o sentido das avaliações.

## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*

Comparando as configurações, o BoW apresentou resultados um pouco melhores que as com TF-IDF. A configuração com BoW - stopwords mantidas teve o maior F1 médio (0,8602), com a BoW - stopwords removidas em seguida (0,8584). Os desvios padrão mostram que houve variação nos resultados entre as divisões de validação cruzada, mas não foram tão altos, com diferenças não tão relevantes.
Escolhi a configuração A (BoW - stopwords mantidas) por apresentar o maior F1 médio. A diferença dos conceitos entre BoW e TF-IDF, é que BoW considera a quantidade de vezes que tal palavra aparece no texto, enquanto TF-IDF considera também a importância de cada palavra em relação aos outros textos.

## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*

A matriz de confusão apresentou 25 negativos e 28 positivos classificados corretamente, e 8 negativos classificados como positivos e 4 positivos classificados como negativos. Para a classe negativa, a precisão foi de 0,86, o recall de 0,76 e o F1 de 0,81. Para a classe positiva, a precisão foi de 0,78, o recall 0,88 e o F1 0,82. 
Para mim, o erro mais grave que uma loja pode cometer é classificar uma reclamação como elogio, pois pode fazer com que os problemas passem despercebidos. 
A acurácia é menos enganosa porque as classes estão equilibradas, com 130 avaliações positivas e 130 negativas.

## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*

A validação cruzada é mais confiável porque avalia cada configuração em cinco divisões diferemtes dos dados, permitindo ver melhor a variação dos resultados. Assim, a comparação não depende apenas de uma só divisão, o que poderia favorecer algum modelo. 
A escolha não deve ser feita olhando o teste, pois deve ser utilizado somente para a avaliação final do modelo escolhido.
O vazamento de dados acontece quando informação dos testes são utilizadas durante o treinamento, e foi evitado no trabalho através da Pipeline, que realiza o pré-processamento e a criação de uma lista dentro de cada divisão da validação cruzada. Assim, aprende o vocabulário apenas com os dados de treinamento.

## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*

Analisando os erros, percebi que o modelo teve dificuldade com frases que dependem do contexnto. Por exemplo, classificou como positiva a avaliação "Panela de péssima qualidade, a comida gruda e queima", mesmo sendo claramente negativa. Também errou a frase "Cheguei animado, mas me arrependi, tudo estava sujo", onde possivelmente pode ter consideraod a palavra "animado" como positiva no geral, sem compreender a mudança de sentimento. Outro erro foi classificar como negativa a frase "Ficou lindo na sala, combinou perfeitamente com os móveis.". Mas palavras influentes, encontrei termos que realmente indicam sentimentos, como "ótimo", "adorei", "detestei" e "ruim". Porém, também apareceram palavras como "bastante", "bem", "dinheiro" e "mais" que não expressam sentimentos por si só. Isso mostra que o modelo aprendeu principalmente associações entre palavras e classes, mas não consegue compreender todos os contextos e os sentidos das frases.

## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

O modelo apresentou dificuldades para interpretar negações, acertando 2 de 4 casos, e nas ironias, acertando 2 de 3. Nas frases mistas, não acertou nenhum dos 2 casos, pois o classificador trabalha apenas com ou positivo ou negativo. Isso acontece porque BoW e TF-IDF consideram principalmente a frequência e a importância das palavras, mas não compreendem completamente o contexto e o sentido das frases. Duas melhorias seriam utilizar n-gramas (*Pesquisei por melhorias e essa realiza combinações de palavras), para considerar combinações de palavras como "não gostei", e experimentar modelos que consigam representam melhor os contexto das frases.
O dataset também possui limitações, como a quantidade pequena de avaliações (260) e a ausência de classes para sentimentos neutros ou mistos. Um risco em produção seria uma reclamação ser classificada como elogio, fazendo com que a empresa deixe de perceber um problema relatado pelos clientes.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [X] Usei IA generativa **somente** para: resumir o que as etapas pediam e pesquisa sobre termos  (não é permitido gerar código ou relatório).
