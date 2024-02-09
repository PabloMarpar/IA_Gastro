import os

def guardar_fragmentos_en_archivos(fragmentos, ruta_salida):
    os.makedirs(ruta_salida, exist_ok=True)
    for i, fragmento in enumerate(fragmentos, 1):
        nombre_archivo = os.path.join(ruta_salida, f"fragmento_{i}.txt")
        with open(nombre_archivo, "w") as archivo:
            archivo.write(" ".join(fragmento))
        print(f"Fragmento {i} guardado en {nombre_archivo}")

def dividir_texto_en_fragmentos(texto, longitud_fragmento):
    palabras = texto.split()
    fragmentos = [palabras[i:i+longitud_fragmento] for i in range(0, len(palabras), longitud_fragmento)]
    return fragmentos

def leer_texto_desde_archivo(ruta_archivo):
    with open(ruta_archivo, "r") as archivo:
        texto = archivo.read()
    return texto

ruta_archivo_original = "/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/tokens.txt"
ruta_salida = "fragmentos_txt"
texto = leer_texto_desde_archivo(ruta_archivo_original)
print(f"Texto original: {texto}")
longitud_fragmento = 500
fragmentos = dividir_texto_en_fragmentos(texto, longitud_fragmento)
guardar_fragmentos_en_archivos(fragmentos, ruta_salida)
