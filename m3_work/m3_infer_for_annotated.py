import twitter
import pandas as pd
from m3inference import M3Twitter
from m3inference import get_lang

m3twitter=M3Twitter()
api = twitter.Api(consumer_key='g9ce9pjY1htdcUhK7A8WTKk9C',
                  consumer_secret='4f1VjkIcHGfpmooVPVA67aTkyY68xWYEIwd1hfzy6p78cYlGED',
                  access_token_key='3010570160-tinLeYvQFXhWm3HaloevmqdQWBmjJM43MVBGaV6',
                  access_token_secret='eq0WO1d1rZtvN9skhHP2tpDGKmh7oBgZEMIuZVfYEzjFi')
annots = pd.read_csv('data\\annotations.tsv', sep='\t')
# remove unannotated
annots = annots[~annots['Gender'].isnull() | ~annots['Gender.1'].isnull()]

m3twitter.transform_jsonl(input_file="data\\twitter_cache\\test.txt",output_file="data\\twitter_cache\\m3_input.jsonl")

for id in annots.user_id:
    u = {}
    user = api.GetUser(id)
    u['id'] = user.id
    u['name'] = user.name
    u['screen_name'] = user.screen_name
    u['description'] = user.description
    # try getting lang as recommended - Tweets get_lang was not implemented
    u['lang'] = get_lang(user.description)
    m3twitter.transform_jsonl()
    print(user)
    print(type(user))
    print(str(user))
    c = m3twitter.infer([str(user)])
    print(c)

    break


