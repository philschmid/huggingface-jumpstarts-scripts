import argparse
import os
import tarfile
import tempfile
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer

dict2class = {
    "text-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="model_id from the model on https://huggingface.co/models")
    parser.add_argument(
        "--task",
        type=str,
        default="text-classification",
        help="model_id from the model on https://huggingface.co/models",
    )
    parser.add_argument("--file_name", type=str, default="pt_model", help="file name of archive")
    parser.add_argument("--output_path", type=str, default=".", help="output path for created archive")

    return parser.parse_args()


def _pack_model(source_dir="", file_name="", target_dir=""):
    source_path = os.path.join(source_dir, f"{file_name}.tar.gz")
    target_path = os.path.join(target_dir, f"{file_name}.tar.gz")
    with tarfile.open(source_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
    os.rename(source_path, target_path)


def main(args):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # load model and save model to temporary directory
        dict2class[args.task].from_pretrained(args.model_id).save_pretrained(tmpdirname)
        AutoTokenizer.from_pretrained(args.model_id).save_pretrained(tmpdirname)
        # pack model into archive
        _pack_model(source_dir=tmpdirname, file_name=args.file_name, target_dir=args.output_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
