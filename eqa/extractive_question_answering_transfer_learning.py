import argparse
import json
import logging
import os
import pathlib
import sys
import tarfile

from constants import constants
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

doc_stride = 384
max_seq_length = 512


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)

    return parser.parse_known_args()


def _get_model_and_tokenizer(args):
    # extract model files
    pretrained_model_path = next(pathlib.Path(args.pretrained_model).glob("*.tar.gz"))
    with tarfile.open(pretrained_model_path) as saved_model_tar:
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
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(saved_model_tar, ".")

    # load model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    return model, tokenizer


def _prepare_data(data_dir, tokenizer):
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        return tokenizer(
            examples[constants.QUESTIONS],
            examples[constants.CONTEXTS],
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    dataset = load_dataset(
        "csv",
        data_files=os.path.join(data_dir, constants.INPUT_DATA_FILENAME),
        column_names=[constants.QUESTIONS, constants.CONTEXTS, constants.ANSWERS_START_POS, constants.ANSWERS_END_POS],
    )["train"]

    dataset = dataset.map(prepare_train_features, batched=True)

    # remove unnecassary columns
    dataset = dataset.rename_column(constants.ANSWERS_START_POS, "start_positions")
    dataset = dataset.rename_column(constants.ANSWERS_END_POS, "end_positions")
    dataset = dataset.remove_columns(["contexts", "questions"])

    # train_test_split dataset
    dataset = dataset.train_test_split(test_size=1 - constants.TRAIN_VAL_SPLIT)
    return dataset["train"], dataset["test"]


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
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        logging_dir=f"{args.model_dir}/logs",
        learning_rate=float(args.adam_learning_rate),
        load_best_model_at_end=True,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
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
