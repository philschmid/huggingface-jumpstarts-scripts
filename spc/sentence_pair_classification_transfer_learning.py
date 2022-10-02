import argparse
import json
import logging
import os
import sys
import tarfile

import boto3
from constants import constants
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)


def download_from_s3(bucket, key, local_rel_dir, model_name):
    local_model_path = os.path.join(os.path.dirname(__file__), local_rel_dir, model_name)
    client = boto3.client("s3")
    client.download_file(bucket, key, local_model_path)
    return


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--model-artifact-bucket", type=str)
    parser.add_argument("--model-artifact-key", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)

    return parser.parse_known_args()


def _prepare_data(data_dir, tokenizer):
    # load dataset from csv
    dataset = load_dataset(
        "csv",
        data_files=os.path.join(data_dir, constants.INPUT_DATA_FILENAME),
        column_names=[constants.LABELS, constants.SENTENCES1, constants.SENTENCES2],
        cache_dir="/opt/ml/input",
    )["train"]

    # preprocess dataset
    preprocessed_dataset = dataset.map(
        lambda batch: tokenizer(
            *(batch[constants.SENTENCES1], batch[constants.SENTENCES2]),
            padding=True,
            max_length=constants.MAX_SEQ_LENGTH,
            truncation=True,
        )
    )
    # train_test_split dataset
    preprocessed_dataset = preprocessed_dataset.train_test_split(test_size=1 - constants.TRAIN_VAL_SPLIT)
    return preprocessed_dataset["train"], preprocessed_dataset["test"]


def _get_model_and_tokenizer(args):
    # download model from s3
    download_from_s3(
        bucket=args.model_artifact_bucket,
        key=args.model_artifact_key,
        local_rel_dir=".",
        model_name=constants.DOWNLOADED_MODEL_NAME,
    )
    # extract model files
    tarball_extract_dir_name = constants.DOWNLOADED_MODEL_NAME.replace(constants.DOT_TAR_GZ, "")
    with tarfile.open(constants.DOWNLOADED_MODEL_NAME) as saved_model_tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner) 
            
        
        safe_extract(saved_model_tar, tarball_extract_dir_name)
    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(tarball_extract_dir_name)
    tokenizer = AutoTokenizer.from_pretrained(tarball_extract_dir_name)
    return model, tokenizer


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_with_args(args):
    model, tokenizer = _get_model_and_tokenizer(args=args)

    train_dataset, eval_dataset = _prepare_data(args.train, tokenizer)

    logging.info(f" loaded train_dataset sizes is: {len(train_dataset)}")
    logging.info(f" loaded eval_dataset sizes is: {len(eval_dataset)}")

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # TODO: is this a good idea?
        evaluation_strategy="epoch",
        logging_dir=f"{args.model_dir}/logs",
        learning_rate=float(args.adam_learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # train model
    trainer.train()

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
