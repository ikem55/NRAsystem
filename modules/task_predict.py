import luigi
from luigi.util import requires

import modules.util as mu
from modules.base_slack import OperationSlack

import pandas as pd
from datetime import datetime as dt

class Sub_get_exp_data(luigi.Task):
    """
    ScikitLearnのモデルで説明変数となるデータを生成するタスク。baoz_intermediateフォルダにデータが格納される
    """
    task_namespace = 'base_predict'
    start_date = luigi.Parameter()
    end_date = luigi.Parameter()
    skproc = luigi.Parameter()
    intermediate_folder = luigi.Parameter()
    export_mode = luigi.Parameter()

    def run(self):
        """
        渡されたexp_data_nameに基づいてSK_DATA_MODELから説明変数のデータを取得する処理を実施。pickelファイル形式でデータを保存
        """
        print("----" + __class__.__name__ + ": run")
        slack = OperationSlack()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start predict job:" + self.skproc.version_str)
        with self.output().open("w") as target:
            print("------ モデル毎に予測データが違うので指定してデータ作成を実行")
            predict_df = self.skproc.create_predict_data()
            print("Sub_get_exp_data run: predict_df", predict_df.shape)
            predict_df.to_pickle(self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl')
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skproc.model_name)


@requires(Sub_get_exp_data)
class End_baoz_predict(luigi.Task):
    """
    ScikitLearnのモデルを元に予測値を計算するLuigiタスク
    """
    task_namespace = 'base_predict'

    def requires(self):
        """
        | 処理対象日ごとにSub_predict_dataを呼び出して予測データを作成する
        """
        print("---" + __class__.__name__+ " : requires")
        return Sub_get_exp_data()

    def run(self):
        """
        | 処理の最後
        """
        print("---" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            exp_data = pd.read_pickle(self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl')
            # 予測を実施
            pred_df = self.skproc.proc_predict_sk_model(exp_data)
            print("End_baoz_predict run: pred_df", pred_df.shape)
            import_df = self.skproc.create_import_data(pred_df)
            if self.export_mode:
                print("export data")
                import_df.to_pickle(self.intermediate_folder + 'export_data.pkl')
                analyze_df = self.skproc.eval_pred_data(import_df)
                print(analyze_df)
            else:
                self.skproc.import_data(import_df)
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish predict job:" + self.skproc.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skproc.model_name)
