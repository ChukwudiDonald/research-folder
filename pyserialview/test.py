import yaml
import pyserialview as psv


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
 
# Access configurations
producer_config = config['producer_config']
consumer_config = config['consumer_config']

psv.view(
    # producer_func=psv.csv_logger,
    producer_func=psv.random_producer,
    consumer_func=psv.running_mean_plotter,
    producer_config=producer_config,
    consumer_config=consumer_config
)


# token = "2yLjVJ6P0oESqedJoAJTxw8LKep_2zDbTTkcsY1wWSmUdeiXP"

# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return {"message": "Hello, Flask server is running!"}
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)


# pip install waitress
# bash
# Copy code
# w