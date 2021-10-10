from itertools import chain
import yaml
import argparse
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def findDiff(d1, d2, stack):
    """
    Compare two dictionaries and return the keys that are different.
    """
    for k, v in d1.items():
        if k not in d2:
            yield stack + ([k] if k else [])
        else:
            if isinstance(v, dict):
                assert isinstance(d2[k], dict)
                for c in findDiff(d1[k], d2[k], [k]):
                    yield stack + c
            else:  # leaf
                if d1[k] != d2[k]:
                    yield stack + [k]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('list_of_files', nargs="*", type=str)
    return parser.parse_args()


def load_file(f):
    e = os.path.splitext(f)[1]
    if e == '.yaml':
        d = yaml.safe_load(open(f, 'rb'))
    else:
        raise Exception(f"I don\'t know how to load extension {e}")
    return d


def get_nested_value(d, keys):
    vitr = d
    for k in keys:
        vitr = vitr[k]
    return vitr


def main(list_of_files):
    all_dicts = {f: load_file(f) for f in list_of_files}
    all_diffs = []
    for i, ci in enumerate(all_dicts.values()):
        for j, cj in enumerate(all_dicts.values()):
            if i == j:
                continue
            all_diffs.append(list(findDiff(ci, cj, [])))
    all_unique_diffs = list(set(chain.from_iterable([tuple(x) for x in diffs]
                                                    for diffs in all_diffs)))
    all_nested_keys_diff = {'-'.join(k): k for k in sorted(all_unique_diffs)}
    for f, d in all_dicts.items():
        print(f"{bcolors.BOLD}{bcolors.OKCYAN}=== {f} ==={bcolors.ENDC}{bcolors.ENDC}")
        for kpretty, k_nested in all_nested_keys_diff.items():
            try:
                v = get_nested_value(d, k_nested)
            except KeyError:
                v = None
            print(f" --- {bcolors.BOLD}{bcolors.WARNING}{kpretty}{bcolors.ENDC}: {v}")


if __name__ == '__main__':
    args = parse_args()
    main(args.list_of_files)
