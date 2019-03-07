import json


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save_namespace(FLAGS, out_path):
    flags_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        json.dump(flags_dict, fp, indent=4, sort_keys=True)


def load_namespace(in_path):
    with open(in_path, 'r') as fp:
        flags_dict = json.load(fp)
    return Bunch(flags_dict)


if __name__ == '__main__':
    flags = load_namespace("quora.sample.config")
    print(flags.train_path)