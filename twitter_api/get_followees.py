import requests
import os
import logging
import argparse
import time
import config


def create_url(user_id):
    return f"https://api.twitter.com/2/users/{user_id}/following"

def get_users_list(csv_location, num_lines_to_skip):
    with open(csv_location) as file_in:
        lines = []
        for line in file_in:
            if num_lines_to_skip > 0:
                num_lines_to_skip -= 1
                continue
            lines.append(line.strip())
    return lines

def check_for_error(user_id, followee_json, output_folder):
    if 'errors' in followee_json:
        error_detail = followee_json['errors'][0]['detail']
        with open(os.path.join(output_folder, "user_errors.csv"), 'a') as outputfile:
            if '[' in error_detail:
                error_detail = error_detail[:error_detail.find('[')] # remove redundant user_id
            outputfile.write(f"{user_id},{error_detail}\n")
        return True
    return False

def check_if_no_followees(followee_json):
    if 'meta' in followee_json:
        if 'result_count' in followee_json['meta']:
            if followee_json['meta']['result_count'] == 0:
                return True
    return False

def get_next_token(followee_json):
    if 'meta' in followee_json:
        if 'next_token' in followee_json['meta']:
            return followee_json['meta']['next_token']
    return None


def write_result_followees(user_id, followee_json, output_folder):
    has_error = check_for_error(user_id, followee_json, output_folder)
    if not has_error:
        if 'data' in followee_json: #has followees
            with open(os.path.join(output_folder, "user_followee_map.csv"), 'a') as outputfile:
                for followee in followee_json['data']:
                    outputfile.write(f"{user_id},{followee['id']}\n")
        elif check_if_no_followees(followee_json):
            with open(os.path.join(output_folder, "users_with_no_followees.csv"), 'a') as outputfile:
                outputfile.write(f"{user_id}\n")
        else:
            print(user_id + ' resulted in and unknown issue, json is: ' + str(followee_json))
    return get_next_token(followee_json)

def get_params(next_token=None):
    if next_token is not None:
        return {"max_results":"1000", "pagination_token": next_token}
    return {"max_results":"1000"}


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
    argparser = argparse.ArgumentParser("Get user followees based on csv of user ids")
    argparser.add_argument("--csv_location", type=str, default='', help="The input csv file")
    argparser.add_argument("--output_folder", type=str, default='', help="The folder to use for output")
    argparser.add_argument("--skip_lines", type=int, default=0, help="Number of lines to skip in input - used for resuming failed jobs")
    args = argparser.parse_args()
    users = get_users_list(args.csv_location, args.skip_lines)
    # init the model object
    count = 1
    for user in users:
        iterate_needed = True
        next_token = None
        while iterate_needed:
            url = create_url(user)
            params = get_params(next_token)
            json_response = connect_to_endpoint(url, params)
            next_token = write_result_followees(user, json_response, args.output_folder)
            if next_token is None:
                iterate_needed = False
        print(F"Finished {count + args.skip_lines} users")
        count += 1

if __name__ == "__main__":
    main()