from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from read_pdf import process, extract_txt_preprocessed


# Charger le modèle et le tokenizer entraînés
model_path = "/home/pdfai/aimodel"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
model.eval()


def get_tags(cv_path):
    text = extract_txt_preprocessed(cv_path)
    text = process(text)

    # Tokenisation
    inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = inputs.pop("offset_mapping")  # (start, end) de chaque token
    with torch.no_grad():
        outputs = model(**inputs)


    # Prédictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    # Récupérer le mapping d'étiquettes
    id2label = model.config.id2label
    entities_by_label = {}
    current_entity_tokens = []
    current_label = None

    for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
        label = id2label[pred_id]
        
        if label == "O":
            if current_label is not None:
                # Convertir la liste des tokens en chaîne avec espaces correctement gérés
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                entities_by_label.setdefault(current_label, []).append(entity_text)
                current_entity_tokens = []
                current_label = None
            continue

        label_prefix, label_type = label.split("-", maxsplit=1)

        if label_prefix == "B":
            if current_label is not None:
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                entities_by_label.setdefault(current_label, []).append(entity_text)
            current_entity_tokens = [token]
            current_label = label_type
        elif label_prefix == "I" and current_label == label_type:
            current_entity_tokens.append(token)
        else:
            # Cas tordu ou erreur → on ferme l'entité en cours
            if current_label is not None:
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
                entities_by_label.setdefault(current_label, []).append(entity_text)
            current_entity_tokens = []
            current_label = None

    # Ajouter la dernière entité si encore ouverte
    if current_label is not None and current_entity_tokens:
        entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens).strip()
        entities_by_label.setdefault(current_label, []).append(entity_text)
    

    return entities_by_label

print(get_tags('/home/pdfai/server/uploads/1718975549573.pdf'))