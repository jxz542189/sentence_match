import csv
import pandas as pd
import os
from zhconv import convert
import codecs


path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, 'data')


def read_csv(file_name, sep='\t', index_col=0, N=None, head=True):
    csv_data = pd.read_csv(os.path.join(data_path, file_name), sep=sep, header=None, index_col=index_col)
    if N == None:
        return csv_data
    else:
        if head:
            return csv_data.head(N)
        else:
            return csv_data.tail(N)


def get_new_data_by_zhconv(file_name, output_filename="atec_new.csv"):
    with codecs.open(os.path.join(data_path, file_name)) as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_words = []
            words = line.split('\t')
            for word in words:
                new_words.append(convert(word, 'zh-cn'))
            new_lines.append(new_words)
    with codecs.open(os.path.join(data_path, output_filename), 'w')  as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(new_lines)


if __name__ == '__main__':
    get_new_data_by_zhconv("atec_nlp_sim_train_all.csv")


    # print(convert('我幹什麼不干你事。', 'zh-cn'))
    # get_new_data_by_zhconv("atec_nlp_sim_train_all.csv")

