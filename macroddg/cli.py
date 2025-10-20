from pathlib import Path
from macroddg import MacroDDG


def main():
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("-i", "--input_json_filepath", type=str, required=True)
    ap.add_argument("-o", "--output_dir", type=str, required=True)
    ap.add_argument("--model_ckpt", type=str, default=str(Path(__file__).parent.parent / "checkpoints"))
    args = ap.parse_args()
    input_json_filepath = args.input_json_filepath
    output_dir = args.output_dir
    model_ckpt = args.model_ckpt
    MacroDDG(input_json_filepath, output_dir, model_ckpt).run()


if __name__ == "__main__":
    main()
