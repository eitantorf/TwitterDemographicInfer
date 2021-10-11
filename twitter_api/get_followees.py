import requests
import os
import logging
import argparse
import time
import config


def create_url(user_id):
    return "https://api.twitter.com/2/users/{}/following".format(user_id)

def get_users_list(csv_location):
    with open(csv_location) as file_in:
        lines = []
        for line in file_in:
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

def check_if_no_followees(user_id, followee_json):
    if 'meta' in followee_json:
        if 'result_cnt' in followee_json['meta']:
            if followee_json['meta']['result_cnt'] == 0:
                return True
    return False


def write_result_followees(user_id, followee_json, output_folder):
    has_error = check_for_error(user_id, followee_json, output_folder)
    if not has_error:
        if 'data' in followee_json: #has followees
            with open(os.path.join(output_folder, "user_followee_map.csv"), 'a') as outputfile:
                for followee in followee_json['data']:
                    outputfile.write(f"{user_id},{followee['id']}\n")
        elif check_if_no_followees(user_id, followee_json, output_folder):
            with open(os.path.join(output_folder, "users_with_no_followees.csv"), 'a') as outputfile:
                outputfile.write(f"{user_id}\n")
        else:
            print(user_id + ' resulted in and unknown issue, json is: ' + str(followee_json))

def get_params():
    return {"max_results":"1000"}
    #return {"user.fields": "created_at"}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {config.bearer_token}"
    r.headers["User-Agent"] = "v2FollowingLookupPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    argparser = argparse.ArgumentParser("Get user followees based on csv of user ids")
    argparser.add_argument("--csv_location", type=str, default='', help="The input csv file")
    argparser.add_argument("--output_folder", type=str, default='', help="The folder to use for output")
    args = argparser.parse_args()
    users = get_users_list(args.csv_location)
    # init the model object
    count = 0

    for user in users:
        if count == 15:
            # rate limiter sleep
            time.sleep(910)
            count = 0
        url = create_url(user)
        params = get_params()
        json_response = connect_to_endpoint(url, params)
        write_result_followees(user, json_response, args.output_folder)
        count += 1

if __name__ == "__main__":
    main()