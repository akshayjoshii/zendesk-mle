# -*- coding: utf-8 -*-

import os
import argparse
from flask import Flask
from intent_classifier import IntentClassifier

app = Flask(__name__)
model = IntentClassifier()


@app.route('/ready')
def ready():
    if model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423


@app.route('/intent')
def intent():
    # Implement this function according to the given API documentation
    pass


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    app.run(port=args.port)
    model.load(args.model)


if __name__ == '__main__':
    main()
