import os
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder

# Укажите имя предобученной модели RuBERT (например, модель от DeepPavlov)
model_name = "DeepPavlov/rubert-base-cased"

# Загрузка токенизатора и модели для классификации
# Укажите количество меток (классов) в вашей задаче
# (Мы определим num_labels после загрузки данных)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка датасета из CSV файлов
# Предполагается, что у вас есть файлы train.csv и validation.csv с колонками "text" и "label"
data_files = {"train": "train.csv", "validation": "validation.csv"}
dataset = load_dataset("csv", data_files=data_files)

# Преобразование строковых меток в числовые
label_encoder = LabelEncoder()
all_labels = dataset["train"]["label"]
label_encoder.fit(all_labels)
num_labels = len(label_encoder.classes_)

def encode_labels(example):
    example["label"] = int(label_encoder.transform([example["label"]])[0])
    return example

dataset = dataset.map(encode_labels)

# Токенизация текстов
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Переименовываем колонку "label" в "labels" (ожидается Trainer-ом)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Загрузка модели с учетом количества меток
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Настройка аргументов для обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Выбор метрики (например, accuracy)
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Создание объекта Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Запуск обучения
trainer.train()

# Сохранение дообученной модели
model.save_pretrained("./finetuned_rubert")
tokenizer.save_pretrained("./finetuned_rubert")
