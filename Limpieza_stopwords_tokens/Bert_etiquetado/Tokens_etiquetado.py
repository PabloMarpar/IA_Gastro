import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW
from tqdm import tqdm
from transformers import BertTokenizer

# Crear el tokenizador BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sintomas = ["Dolor de cabeza", "Mareos", "Problemas de equilibrio",
    "Dolor abdominal intermitente intenso", "Vómitos", "Heces con sangre y mucosidad",
    "Diarrea acuosa", "Deshidratación", "Retraso en el crecimiento",
    "Distensión abdominal severa", "Dolor abdominal", "Fiebre",
    "Sangrado intestinal oculto", "Anemia ferropénica",
    "Confusión mental", "Pérdida de memoria", "Debilidad",
    "Diarrea acuosa profusa", "Fiebre", "Malestar abdominal",
    "Dolor abdominal intenso que irradia hacia la espalda", "Náuseas", "Vómitos",
    "Problemas motores", "Problemas de coordinación", "Hidrocefalia",
    "Sensación de pesadez en el abdomen", "Dolor abdominal", "Náuseas",
    "Pólipos en el colon y el recto", "Sangre en las heces", "Cambios en los hábitos intestinales",
    "Diarrea crónica", "Dolor abdominal", "Pérdida de peso",
    "Cambios en los hábitos intestinales", "Sangre en las heces", "Pérdida de peso",
    "Dolor abdominal en el lado derecho", "Fiebre", "Náuseas",
    "Sangrado intestinal oculto", "Anemia ferropénica", "Heces con sangre",
    "Dolor abdominal", "Sensación de ardor en el estómago", "Náuseas",
    "Diarrea crónica", "Dolor abdominal", "Pérdida de peso",
    "Fatiga", "Dolor abdominal", "Ictericia (coloración amarillenta de la piel y los ojos)",
    "Púrpura palpable", "Dolor abdominal", "Artritis",
    "Dolor en el cuadrante superior derecho del abdomen", "Fiebre", "Dolor al respirar profundamente",
    "Distensión abdominal masiva", "Dolor abdominal", "Náuseas",
    "Diarrea acuosa", "Vómitos", "Fiebre",
    "Sequedad en la boca", "Dificultad para tragar", "Dolor abdominal",
    "Diarrea profusa", "Desnutrición", "Pérdida de peso",
    "Dolor en la parte superior derecha del abdomen", "Náuseas", "Vómitos",
    "Espasmos musculares", "Arqueo del cuerpo", "Reflujo gastroesofágico",
    "Diarrea crónica", "Edema en las extremidades", "Desnutrición",
    "Fatiga", "Hinchazón abdominal", "Pérdida de peso",
    "Retraso en el crecimiento intrauterino", "Baja estatura", "Características faciales distintivas",
    "Edema", "Náuseas", "Pérdida de peso",
    "Dolor en la parte superior derecha del abdomen", "Náuseas", "Vómitos",
    "Ascitis", "Hemorragia gastrointestinal", "Hepatomegalia",
    "Dolor en el cuadrante superior derecho del abdomen", "Fiebre", "Dolor al respirar profundamente",
    "Dolor abdominal recurrente en el cuadrante inferior derecho", "Fiebre", "Náuseas",
    "Retraso en el desarrollo", "Problemas del corazón", "Estreñimiento",
    "Vómitos con sangre", "Dolor abdominal", "Debilidad",
    "Diarrea con sangre", "Dolor abdominal intenso", "Fatiga",
    "Sofocos", "Diarrea intermitente", "Dolor abdominal",
    "Cambios en los hábitos intestinales", "Sangre en las heces", "Pérdida de peso",
    "Problemas de visión", "Pérdida auditiva", "Meningitis" "Dolor abdominal", "Vómitos", "Obstrucción intestinal",
    "Vómitos con sangre", "Debilidad", "Diarrea",
    "Cambios en los hábitos intestinales", "Pérdida de peso",
    "Erupciones cutáneas", "Fiebre", "Dolor articular",
    "Acidez estomacal", "Regurgitación ácida", "Tos crónica",
    "Acidez estomacal crónica", "Tos seca",
    "Ictericia intermitente", "Fatiga",
    "Pérdida de peso", "Cambios en los hábitos intestinales",
    "Pérdida de peso", "Dolor torácico",
    "Diarrea con sangre", "Urgencia para defecar",
    "Distensión abdominal", "Estreñimiento",
    "Dolor abdominal", "Hinchazón",
    "Anemia", "Malformaciones congénitas",  "Diarrea crónica", "Pérdida de peso", "Sensación de saciedad temprana",
    "Náuseas", "Vómitos", "Pólipos en el tracto gastrointestinal",
    "Osteomas", "Tumores cutáneos", "Ascitis",
    "Hepatomegalia", "Dislocaciones articulares", "Deformidades faciales",
    "Problemas cardíacos", "Deshidratación", "Vómitos con sangre",
    "Material similar al café molido", "Diarrea acuosa crónica",
    "Distensión abdominal", "Cambios en los hábitos intestinales",
    "Síncope", "Palpitaciones", "Fatiga", "Amiloidosis gastrointestinal","Convulsiones difíciles de controlar", "Retraso en el desarrollo", "Cambios de comportamiento",
    "Dolor torácico", "Vómitos con contenido alimenticio o sanguinolento", "Fiebre",
    "Problemas de coagulación", "Albinismo", "Enfermedad pulmonar",
    "Ictericia", "Colestasis", "Problemas cardíacos",
    "Pólipos en el tracto gastrointestinal", "Sangrado rectal", "Obstrucción intestinal",
    "Rigidez muscular", "Espasmos musculares", "Ansiedad",
    "Problemas motores", "Problemas de coordinación", "Hidrocefalia",
    "Diarrea crónica", "Dolor abdominal", "Pérdida de peso",
    "Hematomas inexplicables", "Dolor en la piel", "Reacciones emocionales extremas",
    "Estreñimiento crónico", "Hinchazón abdominal", "Malestar",
    "Ictericia", "Fiebre", "Dolor abdominal",
    "Distensión abdominal masiva", "Dolor abdominal", "Náuseas",
    "Hemorragia intestinal", "Obstrucción intestinal", "Protuberancia umbilical",
    "Estreñimiento grave", "Hinchazón abdominal", "Vómitos biliosos",
    "Ictericia", "Fatiga", "Orina oscura",
    "Acidez estomacal severa", "Úlceras gástricas recurrentes", "Diarrea",
    "Falsificación de síntomas médicos", "Búsqueda constante de atención médica", "Historias poco fiables",
    "Dificultad para tragar", "Regurgitación", "Pérdida de peso",
    "Diarrea acuosa", "Dolor abdominal", "Fiebre",
    "Aftas bucales recurrentes", "Úlceras genitales", "Inflamación ocular",
    "Pólipos en el tracto gastrointestinal", "Tumores cutáneos", "Cáncer de tiroides",
    "Náuseas", "Vómitos", "Saciedad temprana",
    "Cambios en los hábitos intestinales", "Sangre en las heces", "Pérdida de peso"

]
enfermedades=["Síndrome de Ogilvie", "Gastroenteritis viral", "Síndrome de Sjögren gastrointestinal",
    "Enfermedad del intestino corto", "Enfermedad del reflujo biliar", "Síndrome de Sandifer",
    "Linfangiectasia intestinal", "Cirrosis hepática", "Síndrome de Silver-Russell",
    "Síndrome de Ménétrier","Cálculos biliares", "Enfermedad de la vena porta", "Síndrome de Fitz-Hugh-Curtis",
    "Síndrome del intestino ciego", "Síndrome de Mowat-Wilson", "Síndrome de Mallory-Weiss",
    "Enfermedad de Crohn", "Síndrome carcinoide", "Síndrome de Lynch",
    "Síndrome de Vogt-Koyanagi-Harada", "Síndrome de la banda inelástica", "Síndrome de Mallory-Weiss",
    "Síndrome de Evans", "Síndrome de Lynch", "Síndrome de intestino corto", "Síndrome de Sweet",
    "Reflujo gastroesofágico (ERGE)", "Enfermedad del reflujo gastroesofágico no erosivo", "Enfermedad de Gilbert",
    "Diverticulitis", "Linfoma intestinal", "Síndrome de intestino corto", "Síndrome de Lynch",
    "Úlcera péptica", "Síndrome de Boerhaave", "Colitis ulcerosa", "Síndrome de Piaget-Lax",
    "Síndrome del intestino irritable (SII)", "Síndrome de Blackfan-Diamond", "Enfermedad de Whipple", "Enteropatía por gluten", "Gastroparesia", "Síndrome de Gardner",
    "Síndrome de Ogilvie", "Síndrome de Peutz-Jeghers", "Enfermedad inflamatoria intestinal (EII)",
    "Síndrome de Budd-Chiari", "Síndrome de Larsen", "Enteropatía por gluten", "Colitis eosinofílica",
    "Colitis microscópica", "Gastropatía por AINES", "Síndrome de Gardner",
    "Poliposis adenomatosa familiar (PAF)", "Síndrome de malabsorción", "Síndrome de Romano-Ward",
    "Síndrome de Romano-Ward", "Amiloidosis gastrointestinal", "Síndrome de Ogilvie",
    "Síndrome de Peutz-Jeghers", "Síndrome de Zollinger-Ellison", "Enfermedad de Wilson",
    "Síndrome de Fitz-Hugh-Curtis", "Enfermedad de Whipple", "Gastroptosis",
    "Enfermedad de Whipple", "Enfermedad de Caroli", "Síndrome de Hirschsprung",
    "Síndrome de Munchausen por poderes", "Polidipsia psicogénica", "Síndrome de McCune-Albright",
    "Dispepsia funcional", "Síndrome de Mowat-Wilson", "Enfermedad de Whipple", "Síndrome carcinoide",
    "Síndrome de Evans","Síndrome de Lennox-Gastaut", "Síndrome de Boerhaave", "Síndrome de Hermansky-Pudlak",
    "Síndrome de Alagille", "Síndrome de Peutz-Jeghers", "Síndrome de Stiff-Person",
    "Síndrome de Dandy-Walker", "Celiaquía", "Síndrome de Gardner-Diamond",
    "Síndrome del intestino perezoso", "Síndrome de Zieve", "Síndrome de Ogilvie",
    "Síndrome de Meckel", "Enfermedad de Hirschsprung", "Síndrome de Dubin-Johnson",
    "Síndrome de Zollinger-Ellison", "Síndrome de Munchausen", "Acalasia", "Enteritis",
    "Enfermedad de Behçet", "Síndrome de Cowden", "Gastroparesia diabética", "Síndrome de Lynch"

]
# Función para tokenizar un síntoma
def tokenizar_sintoma(sintoma, max_length=128):
    tokens = tokenizer.encode_plus(
        sintoma,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    )
    return tokens

# Función para formatear los datos en pares de entrada-salida
def formatear_datos(sintomas, enfermedades, etiquetas_sintomas, etiquetas_enfermedades, tokenizer):
    datos_formateados = []
    for sintoma, enfermedad in zip(sintomas, enfermedades):
        # Tokenizar el síntoma
        tokens = tokenizar_sintoma(sintoma)
        # Obtener la etiqueta correspondiente al síntoma
        etiqueta_sintoma = etiquetas_sintomas[sintoma]
        # Obtener la etiqueta correspondiente a la enfermedad
        etiqueta_enfermedad = etiquetas_enfermedades[enfermedad]
        # Agregar el par de entrada-salida a los datos formateados
        datos_formateados.append((tokens, etiqueta_sintoma, etiqueta_enfermedad))
    return datos_formateados

# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(datos, train_ratio=0.8):
    train_size = int(train_ratio * len(datos))
    train_data = datos[:train_size]
    test_data = datos[train_size:]
    return train_data, test_data

# Asignar un número único a cada síntoma y enfermedad
etiquetas_sintomas = {sintoma: i for i, sintoma in enumerate(set(sintomas))}
etiquetas_enfermedades = {enfermedad: i for i, enfermedad in enumerate(set(enfermedades))}

# Formatear los datos
datos_formateados = formatear_datos(sintomas, enfermedades, etiquetas_sintomas, etiquetas_enfermedades, tokenizer)
# Dividir los datos en conjuntos de entrenamiento y prueba
train_data, test_data = dividir_datos(datos_formateados)

# Verificar el tamaño de los conjuntos de entrenamiento y prueba
print("Tamaño del conjunto de entrenamiento:", len(train_data))
print("Tamaño del conjunto de prueba:", len(test_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convertir los datos en tensores
train_inputs = torch.cat([data[0]['input_ids'] for data in train_data], dim=0)
train_masks = torch.cat([data[0]['attention_mask'] for data in train_data], dim=0)
train_labels = torch.tensor([data[1] for data in train_data])

test_inputs = torch.cat([data[0]['input_ids'] for data in test_data], dim=0)
test_masks = torch.cat([data[0]['attention_mask'] for data in test_data], dim=0)
test_labels = torch.tensor([data[1] for data in test_data])

# Crear conjuntos de datos
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Definir el modelo de BERT para clasificación de secuencias
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(etiquetas_enfermedades))
model.to(device)

# Definir el optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Definir el DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Entrenamiento del modelo
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_loader)
    print(f'Average training loss: {avg_train_loss:.4f}')

# Evaluación del modelo
model.eval()
total_accuracy = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        total_accuracy += (preds == labels).sum().item()

accuracy = total_accuracy / len(test_dataset)
print(f'Accuracy on test set: {accuracy:.4f}')
