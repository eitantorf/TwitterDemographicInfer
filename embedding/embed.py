import pandas as pd
import logging
import argparse
from transformers import BertTokenizer
from transformers import BertModel
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import os
import time

# based on the code from https://databricks.com/blog/2021/10/28/gpu-accelerated-sentiment-analysis-using-pytorch-and-huggingface-on-databricks.html

MODEL = "DeepPavlov/bert-base-cased-conversational"


class TweetLoader(Dataset):
    def __init__(self, text_list=None, tokenizer=None):
        self.tweets = tokenizer(text_list, return_tensors = "pt", padding=True, truncation=True, max_length=512)
        self.tweets = self.tweets['input_ids']

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        data = self.tweets[idx]
        return (data)

def get_file_list(input_location):
    if os.path.isdir(input_location):
        # read all files in the folder
        files = os.listdir(input_location)
        return [os.path.join(input_location, f) for f in files]
    #return just the file
    return [input_location]


def read_and_prep_data(file, tokenizer):
    # check if it's a json or csv and read accoridingly
    extension = os.path.splitext(file)[1]
    if extension == '.json':
        df = pd.read_json(file, lines=True, orient="records")
    elif extension == '.csv':
        df = pd.read_csv(file, names=['id', 'user_id', 'text'], sep='\t')
    else:
        raise Exception("unknown input file type")
    data = TweetLoader(text_list=list(df['text']), tokenizer=tokenizer)
    return df, data

def prep_model(dev):
    model = BertModel.from_pretrained('DeepPavlov/bert-base-cased-conversational', return_dict=True)

    if dev == 'cpu':
        device = torch.device('cpu')
        device_staging = 'cpu:0'
    else:
        device = torch.device('cuda')
        device_staging = 'cuda:0'

    if torch.cuda.device_count() >= 1:
        gpus = list(range(torch.cuda.device_count()))
        print(f"found gpu, list: {gpus}")
    else:
        gpus = [0]

    try:
        model = nn.DataParallel(model, device_ids=gpus)
        model.to(device_staging)
    except:
        torch.set_printoptions(threshold=10000)

    return model, device_staging

def save_to_result_file(initial_df, batch_embedding, batch_size, file_name, output_folder):
    batch_df = initial_df.copy().iloc[:batch_size, ]
    batch_df['embed'] = pd.Series(batch_embedding.cpu().tolist())
    out_filename = os.path.basename(file_name).split('.')[0]
    batch_df[['id', 'user_id', 'embed']].to_csv(os.path.join(output_folder, out_filename + "_embed.csv"),
                                     index=False, header=False, mode='a')

def run_embedding(file, model, device_staging, tokenizer, batch_size, output_folder):
    df, data = read_and_prep_data(file, tokenizer)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)  # Shuffle should be set to False
    t1 = time.time()
    for data in dataloader:
        input = data.to(device_staging)
        with torch.no_grad():
            out = model(input)['pooler_output']
            save_to_result_file(df, out, batch_size, file, output_folder)
            df = df.iloc[batch_size:, ].reset_index(drop=True)
    print(f"finished embedding the file {file}, it took {time.time() - t1} seconds")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    argparser = argparse.ArgumentParser("Embed tweets")
    argparser.add_argument("--input_location", type=str, default='', help="The input file or folder")
    argparser.add_argument("--output_folder", type=str, default='', help="The folder to use for all outputs")
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size to run the inference on")
    argparser.add_argument("--device", type=str, default='cpu', help="cpu is default, use 'gpu' to run on cuda")
    argparser.add_argument("--num_workers", type=int, default=0, help="Num of workers for dataloader")

    args = argparser.parse_args()

    #load the bert tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/bert-base-cased-conversational')
    model, device_staging = prep_model(args.device)

    files = get_file_list(args.input_location)
    for file in files:
        run_embedding(file, model, device_staging, tokenizer, args.batch_size, args.output_folder)
        break



