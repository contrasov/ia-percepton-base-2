import argparse
import gzip
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np


# BASES
SEMENTE = 41
EPOCAS_MLP = 20
PASTA_REF = 'figuras'
AMOSTRAS_TREINO = 55_000
AMOSTRAS_VALIDACAO = 5_000
AMOSTRAS_TESTE = 5_000
DATASET = "mnist"
#DATASET = "fashion-mnist"

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
FASHION_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"


def carregar_dados(nome_dataset):
    """Baixa MNIST ou Fashion-MNIST e devolve imagens achatadas e normalizadas."""
    if nome_dataset == "mnist":
        arquivo = Path("dados/mnist.npz")
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        if not arquivo.exists():
            print("Baixando MNIST (aproximadamente 11 MB)...")
            urlretrieve(MNIST_URL, arquivo)
        with np.load(arquivo) as dados:
            X_treino, y_treino = dados["x_train"], dados["y_train"]
            X_teste, y_teste = dados["x_test"], dados["y_test"]
    elif nome_dataset == "fashion-mnist":
        # Fashion-MNIST é publicado em quatro arquivos IDX compactados, não em NPZ.
        pasta = Path("dados/fashion-mnist")
        pasta.mkdir(parents=True, exist_ok=True)
        base_url = FASHION_MNIST_URL.rstrip("/")
        arquivos = [
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
        ]
        caminhos = []
        for nome in arquivos:
            caminho = pasta / nome
            if not caminho.exists():
                print(f"Baixando Fashion-MNIST: {nome}...")
                urlretrieve(f"{base_url}/{nome}", caminho)
            caminhos.append(caminho)

        with gzip.open(caminhos[0], "rb") as f:
            y_treino = np.frombuffer(f.read(), dtype=np.uint8, offset=8).copy()
        with gzip.open(caminhos[1], "rb") as f:
            X_treino = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28).copy()
        with gzip.open(caminhos[2], "rb") as f:
            y_teste = np.frombuffer(f.read(), dtype=np.uint8, offset=8).copy()
        with gzip.open(caminhos[3], "rb") as f:
            X_teste = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28).copy()
    else:
        raise ValueError(f"Conjunto de dados desconhecido: {nome_dataset}")

    X_treino = X_treino.reshape(-1, 784).astype(np.float32) / 255.0
    X_teste = X_teste.reshape(-1, 784).astype(np.float32) / 255.0
    return X_treino, y_treino.astype(int), X_teste, y_teste.astype(int)


def criar_galeria_amostras(bases):
    """Monta uma galeria com cinco imagens rotuladas de cada base."""
    nomes_fashion = [
        "Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
        "Sandália", "Camisa", "Tênis", "Bolsa", "Bota",
    ]
    fig, eixos = plt.subplots(len(bases), 5, figsize=(12, 5))
    if len(bases) == 1:
        eixos = np.array([eixos])

    rng = np.random.default_rng(SEMENTE)
    for linha, (nome, (X, y)) in enumerate(bases.items()):
        escolhidos = rng.choice(len(y), size=5, replace=False)
        for coluna, indice in enumerate(escolhidos):
            rotulo = int(y[indice])
            texto_rotulo = nomes_fashion[rotulo] if nome == "Fashion-MNIST" else str(rotulo)
            eixo = eixos[linha, coluna]
            eixo.imshow(X[indice].reshape(28, 28), cmap="gray")
            titulo = f"{nome}\nRótulo: {texto_rotulo}" if coluna == 0 else f"Rótulo: {texto_rotulo}"
            eixo.set_title(titulo)
            eixo.axis("off")

    fig.suptitle("Cinco imagens de cada base", fontsize=14)
    fig.tight_layout()
    return fig


class Perceptron:
    """Perceptron multiclasses: uma camada linear e decisão por maior pontuação."""

    def __init__(self, taxa_aprendizado=0.01, epocas=5, seed=SEMENTE):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.seed = seed

    def treinar(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_amostras, n_entradas = X.shape
        n_classes = int(y.max()) + 1
        self.pesos = np.zeros((n_entradas, n_classes))
        self.vies = np.zeros(n_classes)
        historico = []

        for epoca in range(self.epocas):
            for i in rng.permutation(n_amostras):
                previsto = int(np.argmax(X[i] @ self.pesos + self.vies))
                if previsto != y[i]:
                    self.pesos[:, y[i]] += self.taxa_aprendizado * X[i]
                    self.pesos[:, previsto] -= self.taxa_aprendizado * X[i]
                    self.vies[y[i]] += self.taxa_aprendizado
                    self.vies[previsto] -= self.taxa_aprendizado
            erro = float(np.mean(self.prever(X) != y))
            historico.append(erro)
            print(f"  Época {epoca + 1}/{self.epocas} — erro: {erro:.2%}")
        return historico

    def prever(self, X):
        return np.argmax(X @ self.pesos + self.vies, axis=1)


class MLP:
    """Rede totalmente conectada com uma camada oculta ReLU e saída softmax."""

    def __init__(self, entradas=784, ocultos=128, classes=10, taxa_aprendizado=0.1, epocas=EPOCAS_MLP, seed=SEMENTE):
        rng = np.random.default_rng(seed)
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.W1 = rng.normal(0, np.sqrt(2 / entradas), (entradas, ocultos))
        self.b1 = np.zeros(ocultos)
        self.W2 = rng.normal(0, np.sqrt(2 / ocultos), (ocultos, classes))
        self.b2 = np.zeros(classes)

    def prever(self, X):
        oculto = np.maximum(0, X @ self.W1 + self.b1)
        return np.argmax(oculto @ self.W2 + self.b2, axis=1)

    def treinar(self, X, y, epocas=None, tamanho_lote=128):
        total_epocas = self.epocas if epocas is None else epocas
        rng = np.random.default_rng(SEMENTE)
        historico = []
        for epoca in range(total_epocas):
            ordem = rng.permutation(len(X))
            for inicio in range(0, len(X), tamanho_lote):
                indices = ordem[inicio:inicio + tamanho_lote]
                xb, yb = X[indices], y[indices]
                z1 = xb @ self.W1 + self.b1
                h = np.maximum(0, z1)
                logits = h @ self.W2 + self.b2
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                probs[np.arange(len(yb)), yb] -= 1
                probs /= len(yb)

                dW2 = h.T @ probs
                db2 = probs.sum(axis=0)
                dz1 = (probs @ self.W2.T) * (z1 > 0)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)
                lr = self.taxa_aprendizado
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            erro = float(np.mean(self.prever(X) != y))
            historico.append(erro)
            print(f"  Época {epoca + 1}/{total_epocas} — erro: {erro:.2%}")
        return historico


def main():
    parser = argparse.ArgumentParser(description="Treina perceptron e MLP no conjunto escolhido em DATASET.")
    parser.add_argument("--epocas-perceptron", type=int, default=5, help="número de épocas do perceptron")
    args = parser.parse_args()

    dados_escolhidos = carregar_dados(DATASET)
    bases = {
        "MNIST": dados_escolhidos[:2] if DATASET == "mnist" else carregar_dados("mnist")[:2],
        "Fashion-MNIST": dados_escolhidos[:2] if DATASET == "fashion-mnist" else carregar_dados("fashion-mnist")[:2],
    }
    X_treino, y_treino, X_teste, y_teste = dados_escolhidos
    total_treino_validacao = AMOSTRAS_TREINO + AMOSTRAS_VALIDACAO
    if len(X_treino) < total_treino_validacao:
        parser.error("O conjunto de treino não tem imagens suficientes para treino e validação.")
    if len(X_teste) < AMOSTRAS_TESTE:
        parser.error("O conjunto de teste não tem imagens suficientes.")

    # Divide aleatoriamente e de forma reproduzível os 60 mil exemplos de treino.
    indices = np.random.default_rng(SEMENTE).permutation(len(X_treino))
    indices_treino = indices[:AMOSTRAS_TREINO]
    indices_validacao = indices[AMOSTRAS_TREINO:total_treino_validacao]
    X_validacao, y_validacao = X_treino[indices_validacao], y_treino[indices_validacao]
    X_treino, y_treino = X_treino[indices_treino], y_treino[indices_treino]

    # Usa 5 mil exemplos escolhidos de forma reprodutível do conjunto oficial de teste.
    indices_teste = np.random.default_rng(SEMENTE).permutation(len(X_teste))[:AMOSTRAS_TESTE]
    X_teste, y_teste = X_teste[indices_teste], y_teste[indices_teste]

    modelos = {
        "Perceptron": Perceptron(taxa_aprendizado=0.01, epocas=args.epocas_perceptron),
        "MLP": MLP(taxa_aprendizado=0.1, epocas=EPOCAS_MLP),
    }
    resultados = {}
    for nome, modelo in modelos.items():
        print(f"Treinando {nome}...")
        erros = modelo.treinar(X_treino, y_treino)
        acuracia_validacao = float(np.mean(modelo.prever(X_validacao) == y_validacao))
        acuracia_teste = float(np.mean(modelo.prever(X_teste) == y_teste))
        resultados[nome] = (erros, acuracia_validacao, acuracia_teste)
        print(f"{nome}: acurácia na validação = {acuracia_validacao:.2%}")
        print(f"{nome}: acurácia no teste final = {acuracia_teste:.2%}")

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    posicoes = np.arange(len(resultados))
    largura = 0.36
    for indice, (nome, (erros, acuracia_validacao, acuracia_teste)) in enumerate(resultados.items()):
        eixos[0].plot(range(1, len(erros) + 1), erros, marker="o", label=nome)
        eixos[1].bar(posicoes[indice] - largura / 2, acuracia_validacao * 100, largura, color="#f59e0b", label="Validação" if indice == 0 else None)
        eixos[1].bar(posicoes[indice] + largura / 2, acuracia_teste * 100, largura, color="#2563eb", label="Teste final" if indice == 0 else None)
    nome_dataset = "Fashion-MNIST" if DATASET == "fashion-mnist" else "MNIST"
    eixos[0].set(xlabel="Época", ylabel="Erro de treino", title=f"Erro por época — {nome_dataset}")
    eixos[0].grid(alpha=0.3)
    eixos[0].legend()
    eixos[1].set(ylabel="Acurácia (%)", title=f"Acurácia — {nome_dataset}", ylim=(0, 100), xticks=posicoes, xticklabels=list(resultados))
    eixos[1].legend()
    fig.tight_layout()
    pasta_figuras = Path(PASTA_REF)
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    fig_amostras = criar_galeria_amostras(bases)
    arquivo_amostras = pasta_figuras / "amostras_mnist_fashion-mnist.png"
    fig_amostras.savefig(arquivo_amostras, dpi=150, bbox_inches="tight")
    print(f"Galeria de imagens salva em: {arquivo_amostras}")
    arquivo_figura = pasta_figuras / f"comparacao_perceptron_mlp_{DATASET}.png"
    fig.savefig(arquivo_figura, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo em: {arquivo_figura}")
    plt.show()


if __name__ == "__main__":
    main()
