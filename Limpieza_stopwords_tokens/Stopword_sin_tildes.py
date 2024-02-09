import os
import nltk
import re
from nltk.corpus import stopwords
from unidecode import unidecode  # Importar la función unidecode

# Descargar la lista de stopwords en español si no está disponible
nltk.download('stopwords')

# Definir el directorio que contiene los archivos de texto
directorio = '/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/'

# Lista para almacenar el contenido de todos los archivos
contenido_total = []

# Obtener la lista de archivos en el directorio
archivos = os.listdir(directorio)

# Procesar cada archivo en el directorio
for archivo in archivos:
    # Verificar si el archivo es un archivo de texto
    if archivo.endswith('.txt'):
        # Cargar el contenido del archivo
        with open(os.path.join(directorio, archivo), 'r', encoding='utf-8') as file:
            texto = file.read()
            contenido_total.append(texto)

# Combinar todo el contenido en un solo texto
texto_completo = ' '.join(contenido_total)

# Quitar las tildes del texto sin eliminar las letras
texto_sin_tildes = unidecode(texto_completo)

# Tokenizar el texto y convertirlo a minúsculas
tokens = nltk.word_tokenize(texto_sin_tildes.lower())

# Obtener las stopwords en español
stop_words = set(stopwords.words('spanish'))

# Definir una expresión regular para filtrar solo palabras con caracteres alfabéticos
regex = re.compile('[a-zA-Z]+')

# Eliminar stopwords y filtrar palabras por la expresión regular
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and regex.fullmatch(word)]

# Unir las palabras filtradas en un solo texto
filtered_text = ' '.join(filtered_tokens)

# Guardar el texto filtrado en un nuevo archivo
with open(os.path.join(directorio, 'texto_procesado_sintildes.txt'), 'w', encoding='utf-8') as file:
    file.write(filtered_text)

print("Procesamiento completo. Texto filtrado guardado en 'texto_procesado.txt'.")
