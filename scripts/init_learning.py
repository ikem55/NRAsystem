from modules.luigi_tasks import Create_learning_model
from modules.sk_proc import SkProc
import my_config as mc
import sys
from distutils.util import strtobool
import luigi

if __name__ == "__main__":
    args = sys.argv
    test_flag = strtobool(args[1])
    print("tst_flag:" + args[1])

    if test_flag:
        start_date = '2019/01/30'
        end_date = '2019/02/10'
    else:
        start_date = '2015/01/01'
        end_date = '2019/12/31'
    version_sr = "win"
    model_name = "raceuma"
    mock_flag = False
    dict_path = mc.return_base_path(test_flag)
    intermediate_folder = dict_path + 'intermediate/' + model_name + '_' + version_sr + '/'

    skproc = SkProc(version_sr, start_date, end_date, model_name, mock_flag, test_flag)

    luigi.build([Create_learning_model(start_date=start_date, end_date=end_date, skproc=skproc,intermediate_folder=intermediate_folder)],
                local_scheduler=True)