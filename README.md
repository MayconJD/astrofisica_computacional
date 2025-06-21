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

É fundamental para entender a estrutura em larga escala do universo, como as galáxias se agrupam sob a influência da gravidade e da matéria escura, formando filamentos, paredes e aglomerados. A visualização 3D é crucial para explorar essa distribuição complexa.
![image](https://github.com/user-attachments/assets/1b6bd227-3a6e-46bf-a56b-8b4bf156ae4f)

## 2. Simulador Básico de Lentes Gravitacionais Fracas (grav_lens_sim.py)
Este script simula o efeito de distorção que uma massa invisível (como um aglomerado de matéria escura) exerceria sobre a luz de um objeto de fundo (como uma galáxia distante ou uma grade). Ele demonstra como a imagem do objeto de fundo é esticada e curvada à medida que sua luz passa perto da massa da lente.

As lentes gravitacionais são uma ferramenta essencial na astronomia para detectar e mapear a matéria escura, pois é através da sua influência gravitacional na luz que a matéria escura se torna "visível". Este projeto ilustra visualmente esse fenômeno crucial.
![image](https://github.com/user-attachments/assets/3cfd2f69-ea7e-4c9d-85b3-af2809b0437f)

## 3. Classificador Morfológico de Galáxias (Baseado em Características Numéricas) (galaxy_classifier_numerical.py)
Este script treina um modelo de Machine Learning (K-Nearest Neighbors do scikit-learn) para classificar objetos astronômicos do SDSS em três categorias: GALAXY (galáxia), STAR (estrela) e QSO (quasar). A classificação é baseada nas propriedades fotométricas (magnitudes em diferentes filtros de cor e cores derivadas), demonstrando como características numéricas podem ser usadas para a catalogação automatizada.

Essencial para a catalogação em larga escala de objetos astronômicos. Em levantamentos que geram milhões de dados, a classificação manual é inviável. Este projeto mostra como o ML pode automatizar essa tarefa e também ilustra a inerente confusão fotométrica entre certos tipos de objetos (como estrelas e quasares).
![image](https://github.com/user-attachments/assets/5a2f6040-6620-4be7-a093-e0b59abd4618)
![image](https://github.com/user-attachments/assets/1ac42448-aaf0-41e9-9905-281ab1f83d64)

## 4. Analisador de Vizinhança Galáctica (galaxy_neighborhood.py)
Este script calcula a "densidade de vizinhança" para cada galáxia em um catálogo do SDSS, ou seja, quantas galáxias estão próximas dentro de um determinado raio. Em seguida, ele explora a famosa "relação cor-densidade", mostrando como a cor de uma galáxia (relacionada à sua idade e atividade de formação estelar) se correlaciona com a densidade do ambiente em que ela reside.

É fundamental para compreender a evolução das galáxias e como o ambiente local (a presença de outras galáxias, gás e matéria escura) influencia suas propriedades. Revela que galáxias em ambientes densos tendem a ser mais vermelhas e "velhas", enquanto as em ambientes esparsos são mais azuis e "jovens".
![image](https://github.com/user-attachments/assets/b36ec2d3-ca78-4395-acd5-7b080e239e0e)
![image](https://github.com/user-attachments/assets/f7edfecb-9e57-4d6f-82e3-65749862bc69)

## 5. Visualizador 3D de Simulações Cosmológicas Simplificadas (cosmo_sim_viewer.py)
Este script gera um conjunto de dados 3D sintéticos que simulam as grandes estruturas do universo, como aglomerados de galáxias e os filamentos que formam a "teia cósmica" em larga escala. Ele então visualiza essa distribuição de matéria em um espaço 3D interativo.

As simulações são ferramentas essenciais na cosmologia para testar modelos teóricos da formação e evolução do universo. Este projeto oferece uma introdução visual à complexidade da distribuição da matéria, que é moldada pela gravidade ao longo de bilhões de anos.
![image](https://github.com/user-attachments/assets/97c62ddd-3b0b-4c22-b496-11f3617790ac)

## 6. Classificador Morfológico de Galáxias (CNN) (galaxy_classifier_cnn.py)
Este script utiliza o poder das Redes Neurais Convolucionais (CNNs) com TensorFlow/Keras para classificar imagens de galáxias do dataset Galaxy Zoo em tipos morfológicos (Elíptica, Espiral, Irregular). Ele aprende padrões diretamente das imagens para categorizar as galáxias de forma automatizada.

Representa a fronteira da análise automatizada de imagens astronômicas usando Deep Learning. Modelos como este são vitais para classificar as vastas quantidades de dados de imagem geradas por levantamentos modernos e para identificar novas classes de objetos.
![image](https://github.com/user-attachments/assets/1452274a-e85d-4c1a-bfd9-632ddbe8d052)
![image](https://github.com/user-attachments/assets/9e334e0c-7fff-4477-a44f-88aefd35b831)
![image](https://github.com/user-attachments/assets/218c0eb1-598a-429c-9256-f8fbc98c5684)
