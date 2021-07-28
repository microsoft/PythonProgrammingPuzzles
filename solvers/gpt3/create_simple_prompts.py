import json
import re

from run_gpt3_experiments import PARAMS, PREFIX, PREFIX_DOCSTR
import lm_solve

OUTFILENAME = "puzzles_with_prompts.json"


def run(outfilename=OUTFILENAME, filename=PARAMS["filename"]):
    with open(filename, "r") as f:
        entries = json.load(f)

    for entry in entries:
        entry["prompts"] = {}

    for mode in ["short", "medium", "long"]:
        prefix = {
            "short": "",
            "medium": PREFIX,
            "long": PREFIX_DOCSTR
        }[mode]
        prefix = re.sub(r" +$", "", (prefix or "").lstrip(),
                        flags=re.M)  # delete leading/trailing whitespace on each line
        puzzles = lm_solve.load_puzzles(filename, mode == "long")
        prompts = lm_solve.get_prompts(prefix, [f for ft, f in puzzles])
        assert len(puzzles) == len(prompts) == len(entries)
        for entry, prompt in zip(entries, prompts):
            entry["prompts"][mode] = prompt

    with open(outfilename, "w") as f:
        json.dump(entries, f, indent=4)


if __name__ == "__main__":
    run()
