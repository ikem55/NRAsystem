import slackweb
import os
import my_config as mc
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from gspread_dataframe import set_with_dataframe
import requests

class Output(object):
    def __init__(self):
        self.slack_operation_url = mc.SLACK_operation_webhook_url
        self.slack_summary_url = mc.SLACK_summary_webhook_url
        self.slack_realtime_url = mc.SLACK_realtime_webhook_url

    def post_slack_text(self, post_text):
        slack = slackweb.Slack(url=self.slack_operation_url)
        slack.notify(text=post_text)

    def stop_hrsystem_vm(self):
        url = mc.HRsystem_stop_webhook
        response = requests.post(url)
        print(response)

