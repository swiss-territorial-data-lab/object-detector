# see https://realpython.com/command-line-interfaces-python-argparse/#adding-subcommands-to-your-clis
import sys
import argparse
from scripts.generate_tilesets import main as generate_tilesets
from scripts.train_model import main as train_model
from scripts.make_detections import main as make_detections
from scripts.assess_detections import main as assess_detections


def main():

    global_parser = argparse.ArgumentParser(prog="stdl-objdet")

    subparsers = global_parser.add_subparsers(
        title="stages", help="the various stages of the STDL Object Detector Framework"
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

    add_parser = subparsers.add_parser("train_model", help="This script trains an object detection model.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=train_model)

    add_parser = subparsers.add_parser("make_detections", help="This script makes detections, using a previously trained model.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=make_detections)

    add_parser = subparsers.add_parser("assess_detections", help="This script assesses the quality of detections with respect to ground-truth/other labels.")
    add_parser.add_argument(**arg_template)
    add_parser.set_defaults(func=assess_detections)

    # https://stackoverflow.com/a/47440202
    args = global_parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    args.func(*args.operands)


if __name__ == "__main__":

    main()