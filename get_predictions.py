import urllib.request
import json    
from urllib.error import HTTPError
import numpy as np
import argparse
import sys
import pandas as pd

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_features_json(observation):
    features_dict = observation.to_json()
    return features_dict


def get_predictions(body: dict):
    myurl = "http://0.0.0.0:4140/predict"
    jsondata = json.dumps(body, cls=NpEncoder)
    byte_jsondata = jsondata.encode('utf-8')
    headers = {'Content-Type': 'application/json; charset=utf-8',
               'Content-Length': len(byte_jsondata),
               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'}
    req = urllib.request.Request(myurl, headers=headers)
    try:
        response = urllib.request.urlopen(req, byte_jsondata)
        return json.loads(response.read())['predictions']
    except Exception as e:
        print(e.read())
        return str(e)


def main(to_predict):
    features_dict = get_features_json(to_predict)
    preds = get_predictions(features_dict)
    return preds



parser = argparse.ArgumentParser(description='Predict house price')
parser.add_argument('infile', type=str, help='observation.csv')
parser.add_argument('-o', '--outfile', type=str, help='predictions', default='predictions.csv')

args = parser.parse_args()
inputfile = args.infile
outputfile = args.outfile

if not inputfile.endswith('.csv') :
    print('Input file should be present in .csv format')
    sys.exit(1)

test_df = pd.read_csv(inputfile)
assert len(test_df.columns) == 291, "Model expecting 291 features as input data, only % provided" % (len(test_df.columns))

preds = main(test_df)
preds = pd.DataFrame({'id': test_df.id.tolist(), 'price_doc': preds})
preds.to_csv(outputfile, index=False)