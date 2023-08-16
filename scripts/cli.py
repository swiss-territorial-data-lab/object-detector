# see https://realpython.com/command-line-interfaces-python-argparse/#adding-subcommands-to-your-clis

import argparse
from scripts.generate_tilesets import main as generate_tilesets
from scripts.train_model import main as train_model
from scripts.make_predictions import main as make_predictions
from scripts.assess_predictions import main as assess_predictions


def main():

    global_parser = argparse.ArgumentParser(prog="stdl-objdet")

    subparsers = global_parser.add_subparsers(
        title="tasks", help="the various tasks which can be performed by the STDL Object Detector"
    )

    arg_template = {
        "dest": "operands",
        "type": str,
        "nargs": 1,
        "metavar": "<path_to_config_file.yaml>",
        "help": "configuration file",
    }

    add_parser = subparsers.add_parser("generate_tilesets", help="This script generates COCO-annotated training/validation/test/other datasets for object detection tasks.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=generate_tilesets)

    add_parser = subparsers.add_parser("train_model", help="This script trains a predictive model.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=train_model)

    add_parser = subparsers.add_parser("make_predictions", help="This script makes predictions, using a previously trained model.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=make_predictions)

    add_parser = subparsers.add_parser("assess_predictions", help="This script assesses the quality of predictions with respect to ground-truth/other labels.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=assess_predictions)

    args = global_parser.parse_args()

    args.func(*args.operands)


if __name__ == "__main__":

    main()