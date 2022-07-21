import requests
import os
import logging
import argparse
import time
import json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_url(user_ids):
    return f"https://api.twitter.com/2/users?ids={','.join(user_ids)}"

def get_users_list(csv_location, num_lines_to_skip):
    with open(csv_location) as file_in:
        lines = []
        for line in file_in:
            if num_lines_to_skip > 0:
                num_lines_to_skip -= 1
                continue
            lines.append(line.strip())
    return lines

def write_errors_if_exists(response_json, output_folder):
    if 'errors' in response_json:
        with open(os.path.join(output_folder, "user_collection_errors.csv"), 'a') as outputfile:
            for error in response_json['errors']:
                user_id = error['value']
                error_detail = error['detail']
                outputfile.write(f"{user_id},{error_detail}\n")


def write_result_users(response_json, output_folder):
    write_errors_if_exists(response_json, output_folder)
    if 'data' in response_json: #has valid results
        with open(os.path.join(output_folder, "user_details.json"), 'a', encoding='utf8') as outputfile:
            for dic in response_json['data']:
                json.dump(dic, outputfile)
                outputfile.write("\n")

def get_params():
    params = {"user.fields": "created_at,description,id,location,name,profile_image_url,protected,public_metrics,username,url,verified,withheld"}
    return params


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {config.bearer_token}"
    r.headers["User-Agent"] = "v2FollowingLookupPython"
    return r


def connect_to_endpoint(url, params):
    while True:
        response = requests.request("GET", url, auth=bearer_oauth, params=params)
        if response.status_code != 200:
            if response.status_code == 429:
                # rate limiter sleep
                time.sleep(910) # sleep 15 min
            else:
                raise Exception(
                    "Request returned an error: {} {}".format(
                        response.status_code, response.text
                    )
                )
        else:
            return response.json()


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    argparser = argparse.ArgumentParser("Get user details based on csv of user ids")
    argparser.add_argument("--csv_location", type=str, default='', help="The input csv file")
    argparser.add_argument("--output_folder", type=str, default='', help="The folder to use for output")
    argparser.add_argument("--skip_lines", type=int, default=0, help="Number of lines to skip in input - used for resuming failed jobs")
    args = argparser.parse_args()
    users = get_users_list(args.csv_location, args.skip_lines)
    # init the model object
    count = 1
    for i in range(0,len(users),100):
        # work in batches of 100 - since the API supports up to 100
        url = create_url(users[i:i+100])
        params = get_params()
        json_response = connect_to_endpoint(url, params)
        write_result_users(json_response, args.output_folder)
        print(F"Finished {i + 100}")
        count += 1

if __name__ == "__main__":
    main()