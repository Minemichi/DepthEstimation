import requests
import json
# User function
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/.')
from Operation.common import *


class PushSlack():
    def __init__(self, arg_URL, arg_name, arg_id=CHANNEL_ID1, arg_token=ACCESS_TOKEN):
        self.URL = arg_URL
        self.user_name = arg_name
        self.channel_id = arg_id
        self.access_token = arg_token

    def send_text(self, arg_text):
        requests.post(self.URL, data=json.dumps({
            'text': arg_text,  # 投稿するテキスト
            'username': self.user_name,  # 投稿のユーザー名
            'icon_emoji': u':m:',  # 投稿のプロフィール画像に入れる絵文字
            'link_names': 1,  # メンションを有効にする
        }))

    def send_image(self, arg_image):
        files = {'file': open(arg_image, 'rb')}
        param = {'token': self.access_token,
                 'channels': self.channel_id,
                 'filename': "val_predict",
                 'initial_comment': "val_predict",
                 'title': "val_predict"
                 }
        requests.post(url="https://slack.com/api/files.upload", params=param, files=files)
        print("send image.")


if __name__ == '__main__':
    ins_push_slack = PushSlack(CHANNEL_URL1, USER_NAME)
    ins_push_slack.send_text("Hello, world.")
    ins_push_slack.send_image()

