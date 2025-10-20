from pathlib import Path
import re
import json
from macroddg.app.launching.pages import GlobalOptions
from macroddg._version import __version__
from macroddg import MacroDDG
from dp.launching.cli import to_runner, default_minimal_exception_handler


def main(opts: GlobalOptions) -> int:
    input_pdb = opts.input_pdb.get_full_path()
    mutations = re.split("[, \n]+", opts.mutations.strip())
    output_dir = opts.output_dir.get_full_path()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_json_filepath = str(Path(output_dir) / "input.json")
    with open(input_json_filepath, "w") as f:
        json.dump({"pdb_path": input_pdb, "mutcodes": mutations}, f, indent=4)
    MacroDDG(input_json_filepath, output_dir, str(Path(__file__).parent.parent.parent.parent / "checkpoints")).run()
    return 0


def to_parser():
    return to_runner(
        GlobalOptions,
        main,
        version=__version__,
        exception_handler=default_minimal_exception_handler,
    )


if __name__ == "__main__":
    import sys

    to_parser()(sys.argv[1:])
