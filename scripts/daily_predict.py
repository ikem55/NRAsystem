from modules.luigi_tasks import Calc_predict_data
from modules.sk_proc import SkProc
import my_config as mc
import sys
from distutils.util import strtobool
import luigi
from datetime import datetime as dt
from datetime import timedelta

if __name__ == "__main__":
    args = sys.argv
    test_flag = strtobool(args[1])

    if test_flag:
        start_date = '2019/01/30'
        end_date = '2019/02/02'
    else:
        base_start_date = '2020/01/01'
        start_date = SkProc.get_recent_day(base_start_date)
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        ## 過去データがうまく作成できない場合はMDBのレコードを削除する必要がある。
        if start_date > end_date:
            start_date = end_date
    version_sr = "win"
    model_name = "raceuma"
    mock_flag = False
    export_mode = False
    dict_path = mc.return_base_path(test_flag)
    intermediate_folder = dict_path + 'intermediate/' + model_name + '_' + version_sr + '/'

    skproc = SkProc(version_sr, start_date, end_date, model_name, mock_flag, test_flag)

    print("test_flag:" + args[1], " start_date: " + start_date + " end_date: " + end_date)
    luigi.build([Calc_predict_data(start_date=start_date, end_date=end_date, skproc=skproc,intermediate_folder=intermediate_folder, export_mode=export_mode)],
                local_scheduler=True)