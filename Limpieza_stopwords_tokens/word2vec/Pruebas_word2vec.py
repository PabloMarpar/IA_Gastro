from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar el modelo Word2Vec entrenado
model = Word2Vec.load("word2vec_model.model_final")
# Obtener el vocabulario del modelo
vocabulario = model.wv.key_to_index

# Imprimir algunas palabras del vocabulario
print("Vocabulario:")
print(list(vocabulario.keys())[:1000])  # Imprimir las primeras 10 palabras del vocabulario

# 1. Analogías de palabras
print("Analogías de palabras:")
analogy_result = model.wv.most_similar(positive=['estomago', 'ano'], negative=['vitamina'])
print(analogy_result)

# 2. Palabras similares
print("\nPalabras similares a 'colon':")
similar_words = model.wv.most_similar('colon')
print(similar_words)

# 3. Visualización de vectores de palabras
words_of_interest = ['estomago', 'intestino', 'colon', 'cancer', 'ulcera', 'diarrea', 'endoscopia']

# Obtener vectores de palabras y palabras correspondientes
vectors = [model.wv[word] for word in words_of_interest]

# Reducir la dimensión con PCA
pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(vectors)

# Graficar
plt.figure(figsize=(10, 10))
plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1], alpha=0.5)
for i, word in enumerate(words_of_interest):
    plt.annotate(word, xy=(vectors_pca[i, 0], vectors_pca[i, 1]), fontsize=8)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Visualización de vectores de palabras en el contexto de gastroenterología')
plt.show()
