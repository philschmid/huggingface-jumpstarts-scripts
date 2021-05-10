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


class EQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def _get_tokenizer(dir_name):
    config = transformers.AutoConfig.from_pretrained(dir_name)
    org_tokenizer = transformers.AutoTokenizer.from_pretrained(dir_name, config=config)
    fast_tokenizer_class = (
        str(type(org_tokenizer))
        .split(constants.TOKENIZER_CLASS_NAME_SPLIT)[-1]
        .strip(constants.TOKENIZER_CLASS_NAME_STRIP)
        + constants.FAST
    )
    tokenizer = getattr(transformers, fast_tokenizer_class).from_pretrained(dir_name, config=config)
    return config, org_tokenizer, tokenizer


def _add_token_positions(encodings, answers_start_pos, answers_end_pos, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers_start_pos)):
        start_positions.append(encodings.char_to_token(i, answers_start_pos[i]))
        end_positions.append(encodings.char_to_token(i, answers_end_pos[i] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({constants.START_POSITIONS: start_positions, constants.END_POSITIONS: end_positions})


def _prepare_data(data_dir, tokenizer):
    df_data = pd.read_csv(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME),
        header=None,
        names=[constants.QUESTIONS, constants.CONTEXTS, constants.ANSWERS_START_POS, constants.ANSWERS_END_POS],
    )
    df_data = df_data.iloc[np.random.permutation(len(df_data))]
    questions = df_data.copy().pop(constants.QUESTIONS).tolist()
    contexts = df_data.copy().pop(constants.CONTEXTS).tolist()
    ans_start_pos = df_data.copy().pop(constants.ANSWERS_START_POS).tolist()
    ans_end_pos = df_data.copy().pop(constants.ANSWERS_END_POS).tolist()

    _size = int(constants.TRAIN_VAL_SPLIT * len(questions))
    train_questions, val_questions = questions[0:_size], questions[_size:]
    train_contexts, val_contexts = contexts[0:_size], contexts[_size:]
    train_ans_start_pos, val_ans_start_pos = ans_start_pos[0:_size], ans_start_pos[_size:]
    train_ans_end_pos, val_ans_end_pos = ans_end_pos[0:_size], ans_end_pos[_size:]

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    _add_token_positions(train_encodings, train_ans_start_pos, train_ans_end_pos, tokenizer)
    _add_token_positions(val_encodings, val_ans_start_pos, val_ans_end_pos, tokenizer)

    dataset_sizes = {constants.TRAIN: len(train_questions), constants.VAL: len(val_questions)}

    return train_encodings, val_encodings, dataset_sizes


def train_model(dataloaders, dataset_sizes, model, optimizer, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = constants.INITIAL_BEST_LOSS
    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        for phase in [constants.TRAIN, constants.VAL]:
            if phase == constants.TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for batch in dataloaders[phase]:
                input_ids = batch[constants.INPUT_IDS].to(device)
                attention_mask = batch[constants.ATTENTION_MASK].to(device)
                start_positions = batch[constants.START_POSITIONS].to(device)
                end_positions = batch[constants.END_POSITIONS].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == constants.TRAIN):
                    if constants.TOKEN_TYPE_IDS in batch.keys():
                        token_type_ids = batch[constants.TOKEN_TYPE_IDS].to(device)
                        outputs = model(
                            input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions,
                        )
                    else:
                        outputs = model(
                            input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions,
                        )
                    loss = outputs[0]
                    # backward + optimize only if in training phase
                    if phase == constants.TRAIN:
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * input_ids.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            logging.info("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == constants.VAL and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val loss: {:4f}".format(best_loss))

    return best_model_wts


def run_with_args(args):
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
    optimizer = transformers.AdamW(model.parameters(), lr=args.adam_learning_rate)
    config, org_tokenizer, tokenizer = _get_tokenizer(tarball_extract_dir_name)

    train_encodings, val_encodings, dataset_sizes = _prepare_data(args.train, tokenizer)
    train_loader = DataLoader(EQADataset(train_encodings), batch_size=args.batch_size, shuffle=constants.SHUFFLE_TRUE)
    val_loader = DataLoader(EQADataset(val_encodings), batch_size=args.batch_size, shuffle=constants.SHUFFLE_FALSE)
    dataloaders = {constants.TRAIN: train_loader, constants.VAL: val_loader}

    logging.info("dataset sizes: {}".format(dataset_sizes))
    best_model_wts = train_model(dataloaders, dataset_sizes, model, optimizer, num_epochs=args.epochs)

    if args.current_host == args.hosts[0]:
        org_model = torch.load(os.path.join(tarball_extract_dir_name, model_file_name))
        org_model.load_state_dict(best_model_wts)
        torch.save(org_model, os.path.join(args.model_dir, model_file_name))
        org_tokenizer.save_pretrained(args.model_dir)
        config.save_pretrained(args.model_dir)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
