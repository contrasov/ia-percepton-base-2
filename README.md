# Perceptron e MLP com MNIST e Fashion-MNIST

O programa treina e compara um perceptron multiclasses e um MLP em um dos dois conjuntos: MNIST, com imagens de dígitos manuscritos, ou Fashion-MNIST, com imagens de roupas e acessórios. Ambos têm imagens em escala de cinza de 28×28 pixels e dez classes, então os mesmos modelos podem ser usados nos dois. Escolha qual usar editando a variável `DATASET` no começo do `main.py`: deixe `DATASET = "mnist"` ativa ou comente-a e descomente `DATASET = "fashion-mnist"`.

## Pré-requisitos

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e Python 3.10 ou superior.

## Preparar e executar

```bash
uv sync
uv run main.py
```

Na primeira execução, o script baixa os dois conjuntos para a pasta `dados/`, pois mostra uma galeria com cinco imagens rotuladas de cada base. O modelo treina apenas na base escolhida em `DATASET`. Ela é dividida em 55.000 exemplos para treino e 5.000 para validação, de forma reprodutível pela `SEMENTE`. O conjunto oficial de teste, com 10.000 imagens, fica reservado para a avaliação final.

Ao executar, abre a galeria de imagens e o gráfico comparativo. Ambos são salvos na pasta `figuras/`. O MLP treina por 20 épocas. Para ajustar as épocas do perceptron: `uv run main.py --epocas-perceptron 3`.

Todo o código está em `main.py`: carregamento dos dados, implementação dos dois modelos, treinamento, avaliação e gráficos. O perceptron é uma única camada linear; o MLP usa uma camada oculta ReLU e saída softmax. Ambos são implementados diretamente com NumPy para fins de aprendizado.
