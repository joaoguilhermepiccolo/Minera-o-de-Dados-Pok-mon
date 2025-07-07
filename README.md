# Análise de Mineração de Dados - Dataset Pokémon

## Descrição do Projeto
Este projeto aplica técnicas de classificação ao dataset de Pokémon para prever o tipo primário baseado nos atributos de batalha.

## Dataset
- **Fonte:** Hamdalla F. Al-Yasriy / Kaggle
- **Registros:** 1.045 Pokémon
- **Colunas:** 11 atributos incluindo stats de batalha e tipos

## Técnica Aplicada
**Classificação** usando Random Forest para prever o tipo primário do Pokémon baseado em 6 atributos de batalha.

## Principais Resultados
- Acurácia: 23.25% (4x melhor que o acaso)
- Sp.Attack é o atributo mais importante para classificação
- Tipos NORMAL e PSYCHIC são os mais fáceis de prever

## Como Reproduzir
1. Clone o repositório
2. Instale as dependências: `pip install pandas scikit-learn matplotlib seaborn`
3. Execute: `python analise_pokemon.py`

## Arquivos
- `analise_pokemon.py` - Código principal da análise
- `pokemons_dataset.csv` - Dataset utilizado
- `README.md` - Este arquivo

## Tecnologias Utilizadas
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn