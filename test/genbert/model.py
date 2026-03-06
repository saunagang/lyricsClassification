
from datasets import load_dataset,ClassLabel, Dataset, DatasetDict
from transformers import Trainer,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,EarlyStoppingCallback
import numpy as np
from evaluate import load, EvaluationModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

#BASEPATH FOR DATASETS
basepath : str = "balancedDatasets"
folderNames = ["RockPopElectronicFolk", "ClassicalCountryElectronicHip-Hop"]

#LOAD A SPECIFIC DATASET
def dataset_load() -> list[DatasetDict]:
    baseNameDataset = "balancedSubset"
    dataset1 : DatasetDict = load_dataset("csv", data_files= f"{basepath}/{folderNames[0]}/{baseNameDataset}.csv")
    dataset2 : DatasetDict = load_dataset("csv", data_files= f"{basepath}/{folderNames[1]}/{baseNameDataset}.csv")
    return [dataset1,dataset2]

#PRODUCES DATA TUPLES
def getDataStructures(datalist :list[DatasetDict]):
    first = datalist[0], folderNames[0]
    second = datalist[1], folderNames[1] 
    return [first, second]

#RETURNS THE DATA STRUCTURES
def initialiseDataStructures():
    dataList :list[DatasetDict] = dataset_load()
    return getDataStructures(dataList)

#MODEL
model_name : str = "bert-base-uncased"

#DOWNLOAD THE MODEL
def download_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    return model

#INITIALISES THE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function (set : Dataset) :
    return tokenizer(set['lyrics'], padding="max_length", truncation=True,max_length=512)


#THIS FUNCTION CONVERTS THE DATASET SO IT IS APPLICABLE TO BERT

def convertDatasetForBERT(data : DatasetDict) -> Dataset: 
    genres : list = list(set(data["train"]["genre"]))
    #set classlabel
    class_label : ClassLabel = ClassLabel(names=genres)
    #set the classlabel
    dataset : Dataset = data["train"].cast_column("genre", class_label)
    #create split
    dataset_split : Dataset = dataset.train_test_split(test_size=0.2, seed=69, stratify_by_column="genre")
    #tokenize the dataset
    tokenized_dataset : Dataset = dataset_split.map(tokenize_function, batched=True)
    #rename classlabel
    tokenized_dataset = tokenized_dataset.rename_column("genre", "labels")
    return tokenized_dataset

#METRICS USED IN THE TRAINING AND EVALUATION PROCESS

#LOAD METRICS
metric_f1 : EvaluationModule = load("f1")
metric_acc : EvaluationModule = load("accuracy")

#METRICS TO EVALUATE ON
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": metric_acc.compute(
            predictions=preds,
            references=labels
        )["accuracy"],

        "f1_macro": metric_f1.compute(
            predictions=preds,
            references=labels,
            average="macro"  
        )["f1"],
    }
    
    
#INITIALISES TRAINING PARAMETERS WITH GIVEN ARGUMENTS
def initialise_training_arguments(learning_rate : float = 1e-5, label_smoothing_factor : float = 0.05) -> TrainingArguments:
 training_args : TrainingArguments = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",       
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    label_smoothing_factor=label_smoothing_factor,
    fp16=True,
)
 return training_args

#INITIALISES A TRAINER WITH THE GIVEN ARGUMENTS 
def initialiseTrainer(model : str, training_args : TrainingArguments, training_dataset : Dataset) -> Trainer:
    trainer = Trainer(
        model=model,                        
        args=training_args,                 
        train_dataset=training_dataset["train"],
        eval_dataset=training_dataset["test"],   
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    return trainer

#DUMP METRICS FOR FURTHER ANALYSIS
def dump_metrics_to_csv(metrics, basepath: str):
    #CREATE PATH
    out_path = Path(f"{basepath}/Evaluation_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
     json.dump(metrics, f, indent=2)

#PIPELINE FOR EVALUATION
def evaluation_pipeline(trainer : Trainer ,evaluation_set : Dataset,name : str,directory : str = "evaluations"):
    #CREATE DIRECTORY TO SAVE EVALUATION DATA
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    #EVALUATE THE MODEL
    predictions = trainer.predict(evaluation_set['test']) 
    dump_metrics_to_csv(predictions.metrics,f"{directory}/{name}")
    logits = predictions.predictions
    labels = predictions.label_ids
    
    #GET PREDICTIONS 
    preds = np.argmax(logits, axis=-1)
    #CREATE CONFUSION MATRIX AND SAVE 
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.savefig(f"{directory}/{name}Confusion_matrix.png")



#PIPELINE FOR TRAINING 
def training_pipeline() -> Trainer:
    #DOWNLOAD THE MODEL
    model = download_model()
    #DOWNLOAD THE DATASETS
    data_structures = initialiseDataStructures()
    #ITERATE OVER DATASETS
    for dataset in data_structures:
        #TOKENIZE THE DATASET
        tokenised_dataset : Dataset = convertDatasetForBERT(data=dataset[0])
        #CREATE TRAINING ARGUMENTS
        training_arguments : TrainingArguments = initialise_training_arguments()
        #CREATE TRAINER
        trainer : Trainer = initialiseTrainer(model=model, training_args=training_arguments, training_dataset=tokenised_dataset)
        #TRAIN
        trainer.train()
        #SAVE MODEL
        trainer.save_model(f"models/{dataset[1]}model")
        #RUN EVALUATION PIPELINE
        evaluation_pipeline(trainer=trainer,evaluation_set=tokenised_dataset,name=dataset[1])