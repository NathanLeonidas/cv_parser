from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Charger le modèle et le tokenizer entraînés
model_path = "./output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

# Le texte à analyser
text = "ariane chouette ingenieur experiences professionnelles responsable process e-commerce carrefour juillet 2024 aujourdhui contexte de la mission amelioration continue performances drives sur la france entiere analyse process existants et redaction de nouveaux process deplacements recueil des problemes analyse de la situation proposition de plans dactions echange avec les preparateurs les managers et les directeurs magasin tests de solutions analyse des resultats et deploiement des solutions conduisant a des ameliorations accompagnement des equipes sur les nouveaute/changements mis en place suivi de la mise en place des plans dactions et de levolution des kpis chef de projet vie safran aero boosters test cells novembre 2023 mai 2024 contexte de la mission conception installation de bancs dessais moteur avion chez le client travail en milieu international equipes chinoises belges francaises recueil analyse des besoins du client suivi du planning du budget traitement des points bloquants et coordination avec les equipes techniques gestion logistique suivi des envois des passages de douane communication entre equipes techniques/fonctionnelles/clients pack office msp chef de projet stage/cdi engie janvier 2022 avril 2023 contexte de la mission migration de workstation modernisation workplace projet skynote travail en milieu international equipes espagnoles anglaises portugaises recueil analyse des besoins du client gestion des points bloquants urgences priorisation animation et lead des reunions weekly point comites de pilotage gestion logistique suivi des stocks des migrations prise de rendez-vous communication equipes techniques vulgarisation de linformation gestion dequipes techniciens manipulation pack office power bi planification macro-planning formation diplome dingenieur ece paris 2019 2022 specialisation systemes embarques chefferie de projet echange academique semestre alupc lima contact paris france 336-48-26-37-07 chouetteariane@hotmailcom competences soft skills rigueur adaptabilite leadership organisation ecoute communication analyse hard skills logiciels bureautiques java lua c html css power apps dataverse langues anglais avance espagnol avance francais langue maternelle centres dinteret - - - gastronomie sports dequipe voyages /photographie experiences paralleles - - - reserviste armee de lair depuis 2017 service civique bspp 2021 experience aletranger nouvelle-zelande mai 2023 septembre 2023"

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

print(entities_by_label)