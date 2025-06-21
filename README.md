# Estudo de Astrofísica Computacional e Machine Learning com Python

**REPOSITÓRIO DE ESTUDOS SEM AS IMAGENS USADAS NO MACHINE LEARNING E CSV USADO NAS ANÁLISES DEVIDO À LIMITAÇÕES DE TAMANHO**

![Versão](https://img.shields.io/badge/vers%C3%A3o-1.0-blue.svg)
![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Este repositório contém uma coleção de projetos práticos desenvolvidos em Python, focados na aplicação de técnicas de computação e machine learning para explorar e analisar dados no campo da astrofísica. O objetivo é fornecer uma ferramenta de estudo e demonstração para conceitos como a distribuição de matéria no universo, lentes gravitacionais, e a classificação de objetos astronômicos.

---

### Visão Geral dos Projetos

Este estudo abrange diferentes facetas da astrofísica computacional, desde a visualização de grandes estruturas cósmicas até a aplicação de inteligência artificial para categorização de galáxias. Cada projeto é um script Python autônomo que demonstra uma técnica ou conceito específico.

---

## 📖 Sobre o Projeto

A compreensão do universo em que vivemos requer a análise de vastas quantidades de dados observacionais e a simulação de fenômenos complexos. Este projeto nasceu da necessidade de aplicar ferramentas computacionais modernas para auxiliar nessa exploração.

O objetivo é automatizar e visualizar processos complexos, utilizando dados reais de levantamentos astronômicos (como o SDSS - Sloan Digital Sky Survey) e dados simulados. A metodologia é fundamentada em conceitos astronômicos e algoritmos de aprendizado de máquina consolidados.

## ✨ Funcionalidades Abrangentes

Esta coleção de projetos oferece as seguintes funcionalidades:

-   **Visualização 3D de Estruturas Cósmicas:** Mapeamento e visualização de aglomerados de galáxias e estruturas simuladas em três dimensões.
-   **Simulação de Lentes Gravitacionais:** Demonstração visual do efeito de lentes gravitacionais na luz de objetos distantes.
-   **Análise de Densidade de Ambiente Galáctico:** Quantificação da vizinhança de galáxias e sua correlação com propriedades intrínsecas (e.g., cor-densidade).
-   **Classificação de Objetos Astronômicos:** Implementação de modelos de Machine Learning (tradicionais e Deep Learning) para categorizar galáxias, estrelas e quasares com base em suas características.
-   **Processamento e Análise de Grandes Conjuntos de Dados:** Utilização de técnicas para manipulação e extração de informações de catálogos astronômicos.
-   **Fundamentação Científica:** Aplicação de modelos e conceitos validados na astrofísica.

## 🛠️ Tecnologias Utilizadas

-   **Linguagem:** Python 3 (preferencialmente 3.11.x)
-   **Processamento de Imagem:** Pillow, OpenCV
-   **Análise de Dados:** Pandas, NumPy
-   **Aprendizado de Máquina:** scikit-learn, TensorFlow, Keras
-   **Visualização de Dados:** Matplotlib, Seaborn, Plotly

# EXPLICAÇÃO DOS SCRIPTS:

## 1. Explorador Interativo de Aglomerados de Galáxias (galaxy_explorer.py)
Este script carrega dados de galáxias do SDSS, que incluem suas posições celestes (ascensão reta e declinação) e redshift (desvio para o vermelho, indicando distância). Ele converte essas coordenadas para um sistema 3D e, em seguida, utiliza o algoritmo de agrupamento K-Means para identificar e visualizar grupos de galáxias que representam aglomerados cósmicos.

Importância de Estudo: É fundamental para entender a estrutura em larga escala do universo, como as galáxias se agrupam sob a influência da gravidade e da matéria escura, formando filamentos, paredes e aglomerados. A visualização 3D é crucial para explorar essa distribuição complexa.
![image](https://github.com/user-attachments/assets/1b6bd227-3a6e-46bf-a56b-8b4bf156ae4f)

## 2. Simulador Básico de Lentes Gravitacionais Fracas (grav_lens_sim.py)
Este script simula o efeito de distorção que uma massa invisível (como um aglomerado de matéria escura) exerceria sobre a luz de um objeto de fundo (como uma galáxia distante ou uma grade). Ele demonstra como a imagem do objeto de fundo é esticada e curvada à medida que sua luz passa perto da massa da lente.

Importância de Estudo: As lentes gravitacionais são uma ferramenta essencial na astronomia para detectar e mapear a matéria escura, pois é através da sua influência gravitacional na luz que a matéria escura se torna "visível". Este projeto ilustra visualmente esse fenômeno crucial.
![image](https://github.com/user-attachments/assets/3cfd2f69-ea7e-4c9d-85b3-af2809b0437f)
