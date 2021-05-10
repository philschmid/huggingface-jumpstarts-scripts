import argparse
import copy
import json
import logging
import os
import sys
import tarfile
import time

import boto3
import numpy as np
import pandas as pd
import torch
import transformers
from constants import constants
from torch.utils.data import DataLoader


root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

device = torch.device(constants.CUDA_0 if torch.cuda.is_available() else constants.CPU_0)


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


class SPCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _get_tokenizer(dir_name):
    config = transformers.AutoConfig.from_pretrained(dir_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(dir_name, config=config)
    return config, tokenizer


def _prepare_data(data_dir, tokenizer):
    df_data = pd.read_csv(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME),
        header=None,
        names=[constants.LABELS, constants.SENTENCES1, constants.SENTENCES2],
    )
    df_data = df_data.iloc[np.random.permutation(len(df_data))]

    labels = df_data.copy().pop(constants.LABELS).tolist()
    num_labels = np.unique(np.array(labels)).shape[0]
    assert num_labels == 2
    sentences1 = df_data.copy().pop(constants.SENTENCES1).tolist()
    sentences2 = df_data.copy().pop(constants.SENTENCES2).tolist()

    train_data_size = int(constants.TRAIN_VAL_SPLIT * len(labels))
    train_labels, val_labels = labels[0:train_data_size], labels[train_data_size:]
    train_sentences1, val_sentences1 = sentences1[0:train_data_size], sentences1[train_data_size:]
    train_sentences2, val_sentences2 = sentences2[0:train_data_size], sentences2[train_data_size:]

    train_encodings = tokenizer(
        train_sentences1, train_sentences2, max_length=constants.MAX_SEQ_LENGTH, truncation=True, padding=True
    )
    val_encodings = tokenizer(
        val_sentences1, val_sentences2, max_length=constants.MAX_SEQ_LENGTH, truncation=True, padding=True
    )

    dataset_sizes = {constants.TRAIN: len(train_sentences1), constants.VAL: len(val_sentences1)}

    return train_encodings, val_encodings, train_labels, val_labels, dataset_sizes


def train_model(dataloaders, dataset_sizes, model, optimizer, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        for phase in [constants.TRAIN, constants.VAL]:
            if phase == constants.TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for batch in dataloaders[phase]:
                input_ids = batch[constants.INPUT_IDS].to(device)
                attention_mask = batch[constants.ATTENTION_MASK].to(device)
                labels = batch[constants.LABELS].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == constants.TRAIN):
                    if constants.TOKEN_TYPE_IDS in batch.keys():
                        token_type_ids = batch[constants.TOKEN_TYPE_IDS].to(device)
                        outputs = model(
                            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
                        )
                    else:
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    _, preds = torch.max(outputs[1], 1)
                    loss = outputs[0]
                    # backward + optimize only if in training phase
                    if phase == constants.TRAIN:
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logging.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == constants.VAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val acc: {:4f}".format(best_acc))

    return best_model_wts


def _get_model(args):
    download_from_s3(
        bucket=args.model_artifact_bucket,
        key=args.model_artifact_key,
        local_rel_dir=".",
        model_name=constants.DOWNLOADED_MODEL_NAME,
    )
    tarball_extract_dir_name = constants.DOWNLOADED_MODEL_NAME.replace(constants.DOT_TAR_GZ, "")
    with tarfile.open(constants.DOWNLOADED_MODEL_NAME) as saved_model_tar:
        saved_model_tar.extractall(tarball_extract_dir_name)
    model_file_name = args.model_artifact_key.split("/")[1].replace(constants.DOT_TAR_GZ, constants.DOT_PT)
    model = torch.load(os.path.join(tarball_extract_dir_name, model_file_name))
    if torch.cuda.is_available():
        model.to(device)
    model.eval()
    return model, model_file_name, tarball_extract_dir_name


def run_with_args(args):
    model, model_file_name, tarball_extract_dir_name = _get_model(args=args)
    config, tokenizer = _get_tokenizer(tarball_extract_dir_name)
    optimizer = transformers.AdamW(model.parameters(), lr=args.adam_learning_rate)

    train_encodings, val_encodings, train_labels, val_labels, dataset_sizes = _prepare_data(args.train, tokenizer)
    train_loader = DataLoader(
        SPCDataset(train_encodings, train_labels), batch_size=args.batch_size, shuffle=constants.SHUFFLE_TRUE
    )
    val_loader = DataLoader(
        SPCDataset(val_encodings, val_labels), batch_size=args.batch_size, shuffle=constants.SHUFFLE_FALSE
    )
    dataloaders = {constants.TRAIN: train_loader, constants.VAL: val_loader}

    logging.info("dataset sizes: {}".format(dataset_sizes))
    best_model_wts = train_model(dataloaders, dataset_sizes, model, optimizer, num_epochs=args.epochs)

    if args.current_host == args.hosts[0]:
        org_model = torch.load(os.path.join(tarball_extract_dir_name, model_file_name))
        org_model.load_state_dict(best_model_wts)
        torch.save(org_model, os.path.join(args.model_dir, model_file_name))
        tokenizer.save_pretrained(args.model_dir)
        config.save_pretrained(args.model_dir)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
