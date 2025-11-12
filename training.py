from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from evaluate import load
from transformers import DataCollatorForTokenClassification
import evaluate

# Charger le dataset
dataset = load_dataset("json", data_files="./db.json")
dataset = dataset['train'].train_test_split(test_size=0.2)
print(dataset)

categories_tags_no_BI = ["name",
                "location",
                "telephone",
                "mail",
                "languages",
                "interests",
                "skills",
                "entreprises",
                "experience", 
                "academics"
                ]

temp = ['O']
for t in categories_tags_no_BI:
    temp.append('B-'+t)
    temp.append('I-'+t)
categories_tags = temp

id2label = {i: label for i, label in enumerate(categories_tags)}
label2id = {label: i for i, label in id2label.items()}

# Charger le tokenizer et le modèle pré-entraîné
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(categories_tags), id2label=id2label, label2id=label2id
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model.device)

examples = dataset['test']
print(examples)

def tokenize_and_align_labels(example):
    
    tokenized_inputs = tokenizer(example['text'], padding=True, truncation=True) #chaine caractères > tokens
    word_ids = tokenized_inputs.word_ids()  # tokens > numeros de groupe de mots
    word_from_token = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])
    

    labels = [label2id['O']] * len(word_ids)  # catégorie pour chaque token
    
    for ent in example['entities']:
        start, end, label = ent['start'], ent['end'], ent['label']
        
        char_start = tokenized_inputs.char_to_token(start)
        char_end = tokenized_inputs.char_to_token(end - 1)
        
        if char_start is not None and char_end is not None:
            # On ignore le début et la fin de la phrase
            if char_start == 0:
                char_start = None
            if char_end == len(tokenized_inputs['input_ids']) - 1:
                char_end = None

        
        for i, word_id in enumerate(word_ids):
            if word_id is not None:  # Ignore le début et fin
                if char_start is not None and char_end is not None and char_start <= i <= char_end:
                    if i==char_start:
                        labels[i] = label2id['B-'+label]
                    else:
                        labels[i] = label2id['I-'+label]
    tokenized_inputs['labels'] = labels


    if 1==1:
        print(example['text'])
        print(tokenized_inputs['input_ids'])
        print(word_ids)
        print(word_from_token)
        print(labels)
        print(len(labels))
        print('\n')
        for i in range(len(labels)):
            if labels[i]!=-100:
                print(labels[i], word_from_token[i])
    
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=dataset['train'].column_names)


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_dataset['train'][i] for i in range(2)])

metric = evaluate.load('seqeval')

def compute_metrics(eval_preds):
  logits, labels = eval_preds

  predictions = np.argmax(logits, axis=-1)

  true_labels = [[categories_tags[l] for l in label if l!=-100] for label in labels]

  true_predictions = [[categories_tags[p] for p,l in zip(prediction, label) if l!=-100]
                      for prediction, label in zip(predictions, labels)]

  all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

  return {"precision": all_metrics['overall_precision'],
          "recall": all_metrics['overall_recall'],
          "f1": all_metrics['overall_f1'],
          "accuracy": all_metrics['overall_accuracy']}

args = TrainingArguments("camembert-finetuned-ner",
                         eval_strategy = "epoch",
                         save_strategy="epoch",
                         learning_rate = 2e-4,
                         num_train_epochs=100,
                         weight_decay=0.01)

trainer = Trainer(model=model,
                  args=args,
                  train_dataset = tokenized_dataset['train'],
                  eval_dataset = tokenized_dataset['test'],
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)

trainer.train()

trainer.save_model("./output")  # sauvegarde du modèle + config
tokenizer.save_pretrained("./output")  # sauvegarde du tokenizer