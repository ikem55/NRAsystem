import luigi
from luigi.mock import MockTarget
from modules.output import Output
from modules.simulation import Simulation
import modules.util as mu

import os
import pandas as pd
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)
#カラム内の文字数。デフォルトは50だった
pd.set_option("display.max_colwidth", 300)

import pickle
from luigi.util import requires
from datetime import datetime as dt


class Sub_get_learning_data(luigi.Task):
    # 学習に必要なデータを作成する処理。競馬場ごと（各学習用）と全レコード（特徴量作成用）のデータを作成する
    task_namespace = 'base_learning'
    start_date = luigi.Parameter()
    end_date = luigi.Parameter()
    skproc = luigi.Parameter()
    intermediate_folder = luigi.Parameter()

    def run(self):
        # SkModelを読んで学習データを作成する。すべてのデータを作成後、競馬場毎のデータを作成する
        print("----" + __class__.__name__ + ": run")
        slack = Output()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_get_learning_data job:" + self.skproc.version_str)
        with self.output().open("w") as target:
            print("------ learning_dfを作成")
            self.skproc.set_learning_df()
            print("------ 学習用データを保存")
            self.skproc.learning_df.to_pickle(self.intermediate_folder + '_learning.pkl')

            slack.post_slack_text(
                dt.now().strftime("%Y/%m/%d %H:%M:%S") + " finish Sub_get_learning_data job:" + self.skproc.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        # データ作成処理済みフラグを作成する
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + __class__.__name__)

@requires(Sub_get_learning_data)
class Sub_create_feature_select_data(luigi.Task):
    # 特徴量を作成するための処理。target encodingやborutaによる特徴選択の処理を実行
    task_namespace = 'base_learning'

    def requires(self):
        # 前提条件：各学習データが作成されていること
        print("---" + __class__.__name__+ " : requires")
        return Sub_get_learning_data()

    def run(self):
        # 特徴量作成処理を実施。learningの全データ分を取得してSkModel特徴作成処理を実行する
        print("---" + __class__.__name__ + ": run")
        slack = Output()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_create_feature_select_data job:" + self.skproc.version_str)
        with self.output().open("w") as target:
            file_name = self.intermediate_folder + "_learning.pkl"
            with open(file_name, 'rb') as f:
                learning_df = pickle.load(f)
                pd.set_option('display.max_columns', 3000)
                pd.set_option('display.max_rows', 3000)
                print(learning_df.iloc[0])
                print(learning_df.sum())
                self.skproc.create_featrue_select_data(learning_df)
            slack.post_slack_text(dt.now().strftime(
                "%Y/%m/%d %H:%M:%S") + " finish Sub_create_feature_select_data job:" + self.skproc.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        # 処理済みフラグファイルを作成する
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + __class__.__name__)



@requires(Sub_create_feature_select_data)
class Create_learning_model(luigi.Task):
    # BAOZ用の予測モデルを作成する
    # Sub_get_learning_data -> End_baoz_learningの順で実行する
    task_namespace = 'base_learning'

    def requires(self):
        # 学習用のデータを取得する
        print("---" + __class__.__name__+ " : requires")
        return Sub_create_feature_select_data()

    def run(self):
        # 目的変数、場コード毎に学習を実施し、学習モデルを作成して中間フォルダに格納する
        print("---" + __class__.__name__ + ": run")
        slack = Output()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start End_baoz_learning job:" + self.skproc.version_str)
        with self.output().open("w") as target:
            file_name = self.intermediate_folder + "_learning.pkl"
            with open(file_name, 'rb') as f:
                df = pickle.load(f)
                # 学習を実施
                self.skproc.proc_learning_sk_model(df)
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish End_baoz_learning job:" + self.skproc.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))


    def output(self):
        # 学習は何度も繰り返せるようにMockのoutputを返す
        return MockTarget("output")


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
        slack = Output()
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
class Calc_predict_data(luigi.Task):
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
            self.skproc.import_data(import_df)
            slack = Output()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish predict job:" + self.skproc.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skproc.model_name)

@requires(Calc_predict_data)
class Simulate_predict_data(luigi.Task):
    """
    Calc_predict_dataで作成したpredictデータをもとにシミュレーションを行う。
    """

    def requires(self):
        print("---" + __class__.__name__+ " : requires")
        return Calc_predict_data()

    def run(self):
        print("---" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            import_df = pd.read_pickle(self.intermediate_folder + 'export_data.pkl')
            mark_base_df = self.skproc.create_mark_base_df(import_df)
            result_df = self.skproc.get_result_df()
            monthly_mark_base_summary_df = self.skproc.calc_monthly_tanpuku_df(mark_base_df, result_df)
            print(monthly_mark_base_summary_df.sort_values("条件"))

            sim_df = self.skproc.simulate_mark_rate(mark_base_df, result_df)
            print(sim_df)
            target_sr = sim_df.iloc[0]
            print("-------- Mark設定条件 ---------------")
            print(target_sr)
            win_rate = target_sr["win_rate"]
            jiku_rate = target_sr["jiku_rate"]
            ana_rate = target_sr["ana_rate"]

            mark_df = self.skproc.create_mark_df(mark_base_df, win_rate, jiku_rate, ana_rate)
            mark_df.to_pickle(self.intermediate_folder + 'mark_df.pkl')

            summary_df = self.skproc.calc_monthly_mark_df(mark_df, result_df)
            print(summary_df.sort_values("条件"))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skproc.model_name)


@requires(Simulate_predict_data)
class Simulate_kaime_data(luigi.Task):
    """
    Calc_predict_dataで作成したpredictデータをもとにシミュレーションを行う。
    """

    def requires(self):
        print("---" + __class__.__name__+ " : requires")
        return Calc_predict_data()

    def run(self):
        print("---" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            mark_df = pd.read_pickle(self.intermediate_folder + 'mark_df.pkl')
            mock_flag = False
            sim = Simulation(self.start_date, self.end_date, mock_flag, mark_df)

            cond_df = pd.DataFrame()
            for type in ["単勝", "複勝","馬連", "馬単", "ワイド", "三連複"]:
                print(f"----------- {type} -------------")
                proc_sim_sr = sim.proc_fold_simulation(type)
                print(proc_sim_sr)
                if len(proc_sim_sr) != 0:
                    cond = proc_sim_sr["条件"]
                    print("check monthly")
                    sim_df = sim.calc_monthly_sim_df(type, cond)
                    print(sim_df[["年月", "R的中率", "レース数", "件数", "回収率", "払戻偏差", "払戻平均", "的中率", "的中数", "購入総額"]].sort_values(
                        "年月"))
                    proc_sim_sr["タイプ"] = type
                    cond_df = cond_df.append(proc_sim_sr, ignore_index=True)
            print(cond_df)
            cond_df.to_pickle(self.intermediate_folder + 'cond_df.pkl')


    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skproc.model_name)
