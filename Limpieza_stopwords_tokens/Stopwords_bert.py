import os
import re
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
from transformers import BertTokenizer

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

# Obtener las stopwords en español
stop_words = set(stopwords.words('spanish'))

# Definir una expresión regular para filtrar solo palabras con caracteres alfabéticos
regex = re.compile('[a-zA-Z]+')

# Filtrar palabras por la expresión regular y eliminar stopwords
filtered_tokens = [word for word in nltk.word_tokenize(texto_sin_tildes.lower()) if word.lower() not in stop_words and regex.fullmatch(word)]

# Unir las palabras filtradas en un solo texto
filtered_text = ' '.join(filtered_tokens)

# Guardar el texto filtrado en un nuevo archivo
with open(os.path.join(directorio, 'texto_procesado.txt'), 'w', encoding='utf-8') as file:
    file.write(filtered_text)

print("Procesamiento completo. Texto filtrado guardado en 'texto_procesado.txt'.")

# Tokenizar el texto con el tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(filtered_text)

# Convertir los tokens en texto de nuevo
texto_tokenizado_BERT = ' '.join(tokens)

# Guardar el texto tokenizado en un nuevo archivo
with open(os.path.join(directorio, 'texto_tokenizado_BERT.txt'), 'w', encoding='utf-8') as file:
    file.write(texto_tokenizado_BERT)

print("Tokenización completa. Texto tokenizado guardado en 'texto_tokenizado.txt'.")
