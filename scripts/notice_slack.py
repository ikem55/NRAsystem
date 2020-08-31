from modules.report import Report
from modules.output import Output
from modules.import_to_cosmosdb import Import_to_CosmosDB

from datetime import datetime as dt
from datetime import timedelta

n = 0
start_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
end_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
mock_flag = False
output = Output()
rep = Report(start_date, end_date, mock_flag)

post_text = ''
now_time = dt.now()

def export_to_dropbox():
    start_date = dt.now().strftime('%Y/%m') + '/01'
    end_date = dt.now().strftime('%Y/%m/%d')
    rep = Report(start_date, end_date, mock_flag)
    rep.export_bet_df()
    rep.export_race_df()
    rep.export_raceuma_df()


current_text = rep.get_current_text()

if rep.check_flag:
    bet_text = rep.get_todays_bet_text()
    post_text += current_text
    post_text += bet_text

    print(now_time + timedelta(days=1))
    print(rep.final_race_time)
    output.post_slack_real(post_text)
    if now_time > rep.final_race_time:
        print("ok")
        cos = Import_to_CosmosDB(start_date, False)
        cos.import_predict_data()
        target_text = rep.get_kaime_target_text()
        post_text += target_text
        output.post_slack_summary(post_text)
        export_to_dropbox()
        output.stop_hrsystem_vm()

else:
    output.post_slack_real(current_text)