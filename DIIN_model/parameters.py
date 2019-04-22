"""

The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.


"""

import argparse

parser = argparse.ArgumentParser()

# models = ['attmix_CNN', "DIIN"]
# def types(s):
#     options = [mod for mod in models if s in models]
#     if len(options) == 1:
#         return options[0]
#     return s
#
# # Valid genres to train on.
# genres = ['travel', 'fiction', 'slate', 'telephone', 'government']
# def subtypes(s):
#     options = [mod for mod in genres if s in genres]
#     if len(options) == 1:
#         return options[0]
#     return s

pa = parser.add_argument
pa("--batch_size", type=int, default=32, help="batch size") ####
pa("--seq_length", type=int, default=30, help="seq_length")
pa("--emb_dim", type=int, default=100, help="emb_dim")
pa("--hidden_dim", type=int, default=100, help="hidden_dim")

args = parser.parse_args()


