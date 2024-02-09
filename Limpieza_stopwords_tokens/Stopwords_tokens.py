import nltk
import re
from nltk.corpus import stopwords

# Descargar la lista de stopwords en español si no está disponible
nltk.download('stopwords')

# Cargar el archivo de texto tokenizado
with open('/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/tokens.txt', 'r', encoding='utf-8') as file:
    tokenized_text = file.read()

# Tokenizar el texto y convertirlo a minúsculas
tokens = nltk.word_tokenize(tokenized_text.lower())

# Obtener las stopwords en español
stop_words = set(stopwords.words('spanish'))

# Definir una expresión regular para filtrar solo palabras con caracteres alfabéticos
regex = re.compile('[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+')

# Eliminar stopwords y filtrar palabras por la expresión regular
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and regex.fullmatch(word)]

# Unir las palabras filtradas en un solo texto
filtered_text2 = ' '.join(filtered_tokens)

# Guardar el texto filtrado en un nuevo archivo
with open('token_filtered.txt2', 'w', encoding='utf-8') as file:
    file.write(filtered_text2)

print("Stopwords eliminadas y texto filtrado guardado en 'token_filtered.txt'.")
