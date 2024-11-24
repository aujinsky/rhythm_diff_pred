import csv
import requests
import time
import json
import os



def api_fetcher(params = {}, api_key = None):
    try:
        if api_key == None:
            api_key_path = 'utils/osuapi_key.txt'
            api_url = 'https://osu.ppy.sh/api'
            api_key_f = open(api_key_path, 'r')
            api_key = api_key_f.read()
        api_url = api_url+'/get_beatmaps?k='+api_key+'&b='+params['b']
        result = requests.get(api_url).json()[0]
    except:
        return None
    return result

def status_check(result):
    if result["approved"] == "1" or result["approved"] == "4":
        return True
    else:
        return False
