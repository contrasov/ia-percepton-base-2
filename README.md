# Perceptron e MLP com MNIST e Fashion-MNIST

O **Exercício 1** treina e compara um perceptron multiclasses e um MLP em um dos dois conjuntos: MNIST, com imagens de dígitos manuscritos, ou Fashion-MNIST, com imagens de roupas e acessórios. Ambos têm imagens em escala de cinza de 28×28 pixels e dez classes, então os mesmos modelos podem ser usados nos dois. Escolha qual usar editando `DATASET` no começo de `exercicios/exercicio_01/main.py`: deixe `DATASET = "mnist"` ativa ou comente-a e descomente `DATASET = "fashion-mnist"`.

## Pré-requisitos

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e Python 3.10 ou superior.

## Preparar e executar

```bash
uv sync
uv run python exercicios/exercicio_01/main.py
```

Na primeira execução, o exercício baixa os dois conjuntos para a pasta `dados/`, pois mostra uma galeria com cinco imagens rotuladas de cada base. O modelo treina apenas na base escolhida em `DATASET`. Ela é dividida em 55.000 exemplos para treino, 5.000 para validação e 5.000 para teste final, de forma reprodutível pela `SEMENTE`. Os 5.000 exemplos de teste são selecionados do conjunto oficial de teste.

Ao executar, abre a galeria de imagens e o gráfico comparativo. Ambos são salvos na pasta `figuras/`. O MLP treina por 20 épocas. Para ajustar as épocas do perceptron, acrescente `--epocas-perceptron 3` ao comando.

## Organização dos exercícios

- `exercicios/exercicio_01/main.py`: carrega e prepara as bases, mostra amostras rotuladas, separa treino/validação/teste e treina/compara perceptron e MLP.
- Novos exercícios podem ser adicionados em pastas próprias, como `exercicios/exercicio_02/`.

O perceptron é uma única camada linear; o MLP usa uma camada oculta ReLU e saída softmax. Ambos são implementados diretamente com NumPy para fins de aprendizado.
