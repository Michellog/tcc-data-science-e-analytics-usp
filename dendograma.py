import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# 1. Dados mensais dos chamados por módulo (Junho a Setembro de 2026)
dados = {
    'RI':           [7, 11, 9, 5],
    'GL':           [2, 4, 2, 2],
    'AP':           [52, 34, 22, 16],
    'PO':           [26, 22, 30, 8],
    'OM':           [28, 21, 15, 5],
    'iExpenses':    [7, 1, 5, 4],
    'iProcurement': [6, 2, 7, 2]
}

df = pd.DataFrame(dados).T

# 2. Padronização dos dados (Z-score) e Clusterização Hierárquica (Método de Ward)
X_scaled = StandardScaler().fit_transform(df)
Z = linkage(X_scaled, method='ward')

# 3. Construção do Dendrograma
plt.figure(figsize=(10, 5))
dendrogram(
    Z,
    labels=df.index,
    color_threshold=1.5,
    above_threshold_color='grey',
    leaf_rotation=35
)

# 4. Elementos visuais (Linha de corte, Rótulos e Título)
plt.axhline(y=1.5, color='tab:blue', linestyle='--', label='Ponto de corte (1,5)')
plt.title('Dendrograma da clusterização hierárquica dos principais módulos EBS', fontsize=11)
plt.xlabel('Módulos EBS')
plt.ylabel('Distância Euclidiana')
plt.legend(loc='upper right')
plt.tight_layout()

# 5. Salvar a figura para o TCC
plt.savefig('dendrograma_tcc.png', dpi=300)
plt.show()
