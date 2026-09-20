"""Command-line entry point: `trustmed <command> [options]`."""
import argparse

from trustmed import config


def main():
    parser = argparse.ArgumentParser(prog="trustmed", description="Trustworthy blood-cell classification.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--size", type=int, default=config.IMAGE_SIZE, choices=[28, 64, 128, 224],
                             help="image size in pixels")
        command.add_argument("--limit", type=int, default=None,
                             help="use only this many images per split, for a quick test")
        return command

    add("explore", "count images per class and save example pictures")
    train_cmd = add("train", "fine-tune one ResNet-18")
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--epochs", type=int, default=config.EPOCHS)
    add("predict", "save the predictions of every trained model, clean and shifted")
    add("evaluate", "turn saved predictions into metrics, tables and charts")
    explain_cmd = add("explain", "Grad-CAM heatmaps and their sanity check")
    explain_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Imports happen here, not at the top, so `trustmed --help` answers instantly
    if args.command == "explore":
        from trustmed.explore import explore
        explore(args.size, args.limit)
    elif args.command == "train":
        from trustmed.train import train
        train(args.size, args.seed, args.epochs, args.limit)
    elif args.command == "predict":
        from trustmed.predict import predict
        predict(args.size, args.limit)
    elif args.command == "evaluate":
        from trustmed.evaluate import evaluate
        evaluate(args.size, args.limit)
    elif args.command == "explain":
        from trustmed.explain import explain
        explain(args.size, args.seed, args.limit)
