from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Descargar las listas de stopwords en español e inglés si no están disponibles
nltk.download('stopwords')
nltk.download('punkt')

# Cargar el archivo de texto tokenizado
with open('/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/texto_procesado_sintildes.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    sentences = sent_tokenize(text)

# Tokenización de palabras y preprocesamiento de cada oración
tokenized_text = []
for sentence in sentences:
    # Tokenización de palabras y eliminación de caracteres no deseados
    words = word_tokenize(sentence)
    words = [re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚ]', '', word) for word in words]
    words = [word.lower() for word in words if
             word.isalnum()]  # Convertir a minúsculas y eliminar palabras no alfanuméricas

    # Eliminar stopwords en español e inglés
    stop_words_es = set(stopwords.words('spanish'))
    stop_words_en = set(stopwords.words('english'))
    stop_words = stop_words_es.union(stop_words_en)
    words = [word for word in words if word not in stop_words]

    tokenized_text.append(words)

# Entrenamiento del modelo Word2Vec
model = Word2Vec(sentences=tokenized_text, vector_size=400, window=50, min_count=5, workers=8)
print("Tamaño del vocabulario:", len(model.wv))

model.save("word2vec_model.model_final")
