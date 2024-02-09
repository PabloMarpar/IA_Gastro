from transformers import BertTokenizer, BertModel
import torch

# Cargar el tokenizador y el modelo preentrenado de ClinicalBERT
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Función para identificar y clasificar entidades médicas importantes en un texto
def identificar_entidades_medicas(texto):
    # Lista para almacenar las entidades médicas identificadas
    entidades_medicas = []

    # Dividir el texto en fragmentos más pequeños de máximo 512 tokens (longitud máxima admitida por ClinicalBERT)
    max_length = 512
    fragmentos = [texto[i:i + max_length] for i in range(0, len(texto), max_length)]

    # Procesar cada fragmento del texto
    for fragmento in fragmentos:
        # Tokenizar el fragmento de texto
        tokens = tokenizer.tokenize(fragmento)

        # Convertir los tokens en ids numéricos
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Agregar tokens de inicio y fin especiales para el modelo
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # Convertir a tensor
        input_ids = torch.tensor([input_ids])

        # Obtener las representaciones ocultas del modelo ClinicalBERT
        with torch.no_grad():
            outputs = model(input_ids)

        # Esto requerirá un paso adicional para procesar las salidas del modelo
        # Dependiendo de cómo estén estructuradas las salidas de ClinicalBERT
        # y cómo se realizó el entrenamiento para identificar las entidades médicas

        # Agregar las entidades médicas identificadas a la lista
        # entidades_medicas.extend(entidades_identificadas)

    return entidades_medicas


# Función para leer el texto desde un archivo
def leer_archivo(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        texto = archivo.read()
    return texto


# Ruta del archivo de texto
ruta_archivo = "/home/pablo/Modelo_gastro/Limpieza_stopwords_tokens/tokens.txt"

# Leer el texto desde el archivo
texto = leer_archivo(ruta_archivo)

# Identificar y clasificar entidades médicas importantes en el texto
entidades_medicas = identificar_entidades_medicas(texto)

# Imprimir las entidades médicas identificadas
print("Entidades médicas identificadas:")
for entidad in entidades_medicas:
    print("-", entidad)
