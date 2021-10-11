import argparse
import pandas as pd
import os
from m3inference.m3twitter import M3Twitter
import logging
import time
import m3inference

def check_is_default_profile_image(image_url):
    if image_url == "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png":
        return True
    return False

def parse_csv_into_json(csv_path, output_folder, num_rows, starting_point):
    '''
    changing the input csv from achtung to the USER json object m3 needs
    :param csv_path:
    :return:
    '''
    start = time.time()
    with open(csv_path, 'r', encoding="utf8") as file:
        header = file.readline().strip().split('\t')
    # load the file with the header - skip rows as needed and only load num_rows rows
    input_df = pd.read_csv(csv_path, sep='\t', names=header, skiprows=starting_point*num_rows, nrows=num_rows, header=0)
    if input_df.empty:
        # we passed through all files, return False
        return False
    input_df = input_df.rename(columns={"uid": "id_str"})
    # convert id to str
    input_df['id_str'] = input_df['id_str'].astype(str)
    # add the needed deafult profile image flag
    input_df['default_profile_image'] = input_df.profile_image_url_https.apply(lambda x: check_is_default_profile_image(x))
    # leave only needed columns
    input_df = input_df[['id_str','profile_image_url_https','description','name','screen_name','default_profile_image']]
    input_df.to_json(os.path.join(output_folder, "input_users.jsonl"), orient="records", lines=True)

    return True

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    argparser = argparse.ArgumentParser("M3 infer based on csv")
    argparser.add_argument("--csv_location", type=str, default='', help="The input csv file")
    argparser.add_argument("--output_folder", type=str, default='', help="The folder to use for all outputs")
    argparser.add_argument("--batch_size", type=int, default=10, help="Batch size to run the inference on")
    args = argparser.parse_args()
    first_batch = True
    rows_left = True
    num_batches = 0
    # init the model object
    m3twitter = M3Twitter(cache_dir=os.path.join(args.output_folder, "twitter_cache"))
    while (rows_left):
        start = time.time()
        rows_left = parse_csv_into_json(args.csv_location, args.output_folder, args.batch_size, num_batches)
        if not rows_left:
            # no rows returned - gone throgh all file
            break
        logging.info(f"Loading input csv with took {time.time() - start} seconds")
        start = time.time()
        m3twitter.transform_jsonl(input_file=os.path.join(args.output_folder, "input_users.jsonl"), output_file=os.path.join(args.output_folder, "m3_input.jsonl"))
        logging.info(f"Finished transform_jsonl in  {time.time() - start} seconds")
        start = time.time()
        result_df = m3twitter.infer(os.path.join(args.output_folder, "m3_input.jsonl"), output_format='dataframe', num_workers=1, batch_size=args.batch_size)
        logging.info(f"Finished inference in  {time.time() - start} seconds")
        if first_batch:
            # first time writing the output file - w mode
            result_df.to_csv(os.path.join(args.output_folder, "inference_results.csv"), index=False)
        else:
            # append mode
            result_df.to_csv(os.path.join(args.output_folder, "inference_results.csv"), index=False, mode='a', header=False)
        num_batches += 1
        first_batch = False
