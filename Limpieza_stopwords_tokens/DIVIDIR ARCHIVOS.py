import os


def dividir_archivo(ruta_archivo, parte1, parte2):
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        contenido = file.read()

    longitud_total = len(contenido)
    mitad = longitud_total // 2

    # Dividir el contenido en dos partes
    contenido_parte1 = contenido[:mitad]
    contenido_parte2 = contenido[mitad:]

    # Escribir la primera parte en un archivo
    with open(parte1, 'w', encoding='utf-8') as file:
        file.write(contenido_parte1)

    # Escribir la segunda parte en otro archivo
    with open(parte2, 'w', encoding='utf-8') as file:
        file.write(contenido_parte2)


# Rutas de los archivos
archivo_original = '/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/texto_procesado.txt'
parte1 = '/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/TextoWordvec1.txt'
parte2 = '/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/TextoWord2vec2.txt'

# Llamar a la función para dividir el archivo
dividir_archivo(archivo_original, parte1, parte2)

print("El archivo se ha dividido en dos partes y se han guardado en", parte1, "y", parte2)
