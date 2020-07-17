from modules.extract import Extract
from modules.transform import Transform as Tf
import modules.util as mu
import my_config as mc

from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

class Load(object):
    """
    データロードに関する共通処理を定義する。
    race,raceuma,prev_raceといった塊ごとのデータを作成する。learning_df等の最終データの作成はsk_proc側に任せる

    """
    dict_folder = ""
    """ 辞書フォルダのパス """
    mock_flag = False
    """ mockデータを利用する場合はTrueにする """
    race_df = ""
    raceuma_df = ""
    horse_df = ""
    prev_raceuma_df = ""
    grouped_raceuma_prev_df = ""
    result_df = ""

    def __init__(self, version_str, start_date, end_date, mock_flag, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.dict_path = mc.return_base_path(test_flag)
        self._set_folder_path(version_str)
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)
        self.tf = self._get_transform_object(start_date, end_date)

    def _set_folder_path(self, version_str):
        self.dict_folder = self.dict_path + 'dict/' + version_str + '/'
        print("self.dict_folder:", self.dict_folder)

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Extract(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def set_race_df(self):
        """  race_dfを作成するための処理。race_dfに処理がされたデータをセットする """
        race_base_df = self.ext.get_race_before_table_base()
        self.race_df = self._proc_race_df(race_base_df)
        print("set_race_df: race_df", self.race_df.shape)

    def _proc_race_df(self, race_df):
        """ race_dfの前処理、encode -> normalize -> standardize -> feature_create -> drop_columnsの順で処理 """
        race_df = self.tf.encode_race_df(race_df)
        race_df = self.tf.normalize_race_df(race_df)
        race_df = self.tf.standardize_race_df(race_df)
        race_df = self.tf.create_feature_race_df(race_df)
        race_df = self.tf.choose_race_df_columns(race_df)
        return race_df

    def set_raceuma_df(self):
        """ raceuma_dfを作成するための処理。raceuma_dfに処理がされたデータをセットする """
        raceuma_base_df = self.ext.get_raceuma_before_table_base()
        self.raceuma_df = self._proc_raceuma_df(raceuma_base_df)
        print("set_raceuma_df: raceuma_df", self.raceuma_df.shape)

    def _proc_raceuma_df(self, raceuma_df):
        raceuma_df = self.tf.encode_raceuma_df(raceuma_df, self.dict_folder)
        raceuma_df = self.tf.normalize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.standardize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_df(raceuma_df)
        raceuma_df = self.tf.choose_raceuma_df_columns(raceuma_df)
        return raceuma_df.copy()

    def set_horse_df(self):
        """  horse_dfを作成するための処理。horse_dfに処理がされたデータをセットする """
        horse_base_df = self.ext.get_horse_table_base()
        self.horse_df = self._proc_horse_df(horse_base_df)
        print("set_horse_df: horse_df", self.horse_df.shape)

    def _proc_horse_df(self, horse_df):
        horse_df = self.tf.encode_horse_df(horse_df)
        horse_df = self.tf.normalize_horse_df(horse_df)
        horse_df = self.tf.standardize_horse_df(horse_df)
        horse_df = self.tf.create_feature_horse_df(horse_df)
        horse_df = self.tf.choose_horse_df_column(horse_df)
        return horse_df.copy()

    def set_prev_df(self, race_df, raceuma_df):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        race_result_df, raceuma_result_df = self._get_prev_base_df(5)
        prev_raceuma_result_df = self.tf.drop_prev_raceuma_result_df_columns(raceuma_result_df)
        prev5_raceuma_df = self._get_prev_df(5, race_result_df, prev_raceuma_result_df, raceuma_df)
        prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
        prev5_raceuma_df.rename(columns={"競走コード_5": "競走コード", "馬番_5": "馬番"}, inplace=True)
        prev4_raceuma_df = self._get_prev_df(4, race_result_df, prev_raceuma_result_df, raceuma_df)
        prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
        prev4_raceuma_df.rename(columns={"競走コード_4": "競走コード", "馬番_4": "馬番"}, inplace=True)
        prev3_raceuma_df = self._get_prev_df(3, race_result_df, prev_raceuma_result_df, raceuma_df)
        prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
        prev3_raceuma_df.rename(columns={"競走コード_3": "競走コード", "馬番_3": "馬番"}, inplace=True)
        prev2_raceuma_df = self._get_prev_df(2, race_result_df, prev_raceuma_result_df, raceuma_df)
        prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        prev2_raceuma_df.rename(columns={"競走コード_2": "競走コード", "馬番_2": "馬番"}, inplace=True)
        prev1_raceuma_df = self._get_prev_df(1, race_result_df, prev_raceuma_result_df, raceuma_df)
        prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        prev1_raceuma_df.rename(columns={"競走コード_1": "競走コード", "馬番_1": "馬番"}, inplace=True)
        prev_raceuma_df = pd.merge(prev1_raceuma_df, prev2_raceuma_df, on=["競走コード", "馬番"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev3_raceuma_df, on=["競走コード", "馬番"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev4_raceuma_df, on=["競走コード", "馬番"], how="outer")
        prev_raceuma_df = pd.merge(prev_raceuma_df, prev5_raceuma_df, on=["競走コード", "馬番"], how="outer")
        self.prev_raceuma_df = prev_raceuma_df
        self._set_grouped_raceuma_prev_df(race_result_df, raceuma_result_df, race_df, raceuma_df)

    def _get_prev_base_df(self, num):
        """ 過去データを計算するためのベースとなるDataFrameを作成する

        :param int num: int(計算前走数）
        :return: dataframe
        """
        print("_get_prev_base_df" + str(num))
        dt_start_date = dt.strptime(self.start_date, '%Y/%m/%d')
        prev_start_date = (dt_start_date - timedelta(days=(180 + 60 * int(num)))).strftime('%Y/%m/%d')
        ext_prev = self._get_extract_object(prev_start_date, self.end_date, self.mock_flag)
        race_base_df = ext_prev.get_race_table_base()
        race_base_result_df = self._proc_race_result_df(race_base_df)
        raceuma_base_df = ext_prev.get_raceuma_table_base()
        raceuma_base_df = raceuma_base_df.query("異常区分コード == 0").copy()
        raceuma_base_result_df = self._proc_raceuma_result_df(raceuma_base_df)
        race_winner_df = self._get_race_winner_df(raceuma_base_result_df)
        race_result_df = self.tf.create_feature_race_result_winner_df(race_base_result_df, race_winner_df)
        raceuma_result_df = self.tf.create_feature_raceuma_result_race_df(race_base_result_df, raceuma_base_result_df)
        race_result_df = self.tf.drop_race_result_df_columns(race_result_df)
        raceuma_result_df = self.tf.drop_raceuma_result_df_columns(raceuma_result_df)
        return race_result_df, raceuma_result_df

    def _proc_race_result_df(self, race_df):
        """ race_dfの前処理、encode -> normalize -> standardize -> feature_create -> drop_columnsの順で処理 """
        race_df = self.tf.encode_race_result_df(race_df)
        race_df = self.tf.normalize_race_result_df(race_df)
        race_df = self.tf.standardize_race_result_df(race_df)
        race_df = self.tf.create_feature_race_result_df(race_df)
        race_df = self.tf.choose_race_result_df_columns(race_df)
        return race_df

    def _proc_raceuma_result_df(self, raceuma_df):
        raceuma_df = self.tf.encode_raceuma_result_df(raceuma_df, self.dict_folder)
        raceuma_df = self.tf.normalize_raceuma_result_df(raceuma_df)
        raceuma_df = self.tf.standardize_raceuma_result_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_result_df(raceuma_df)
        raceuma_df = self.tf.choose_raceuma_result_df_columns(raceuma_df)
        return raceuma_df.copy()

    def _get_race_winner_df(self, raceuma_base_df):
        race_winner_df = raceuma_base_df[raceuma_base_df["確定着順"] == 1].drop_duplicates(subset='競走コード')
        return race_winner_df

    def _get_prev_df(self, num, race_result_df, raceuma_result_df, raceuma_df):
        """ numで指定した過去走のデータを取得して、raceuma_base_df,race_base_dfにセットする。競走コードと馬番は今回のものがセットされる

        :param int num: number(過去１走前の場合は1)
        """
        prev_race_key = "近走競走コード" + str(num)
        prev_umaban = "近走馬番" + str(num)
        raceuma_base_df = raceuma_df[["競走コード", "馬番", prev_race_key, prev_umaban]]
        temp_prev_raceuma_df = raceuma_result_df.rename(columns={"競走コード": prev_race_key, "馬番": prev_umaban})
        this_raceuma_df = pd.merge(raceuma_base_df, temp_prev_raceuma_df, on=[prev_race_key, prev_umaban])
        this_raceuma_df = this_raceuma_df.drop([prev_race_key, prev_umaban], axis=1)

        race_base_df = raceuma_base_df[["競走コード", "馬番", prev_race_key]]
        temp_prev_race_df = race_result_df.rename(columns={"競走コード": prev_race_key})
        this_race_df = pd.merge(race_base_df, temp_prev_race_df, on=prev_race_key)
        this_race_df = this_race_df.rename(columns={"競走コード_x": "競走コード"}).drop(prev_race_key, axis=1)
        merged_df = pd.merge(this_race_df, this_raceuma_df, on=["競走コード", "馬番"])
#        merged_df = merged_df.drop(['タイム指数', '単勝オッズ', '先行率', 'ペース偏差値', '距離増減', '斤量比', '追込率', '平均タイム',
#             "距離", "頭数", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち",
#             "年月日", "月日", "距離", "血統登録番号"], axis=1)

        return merged_df


    def _set_grouped_raceuma_prev_df(self, race_result_df, raceuma_result_df, race_df, raceuma_df):
        """  過去走の集計データを作成する。開始日の1年前のデータを取得して条件毎の集計データを作成する

        :return: dataframe
        """
        # レースデータを取得
        race_base_df = race_df[["競走コード", "場コード", "距離", "ナイター", "季節", "非根幹", "距離グループ", "頭数グループ"]].copy()

        # レース馬データを取得
        raceuma_base_df = raceuma_df[["競走コード", "馬番", "血統登録番号", "年月日", "馬番グループ", "距離増減"]].copy()

        # レースとレース馬データを結合したデータフレーム（母数となるテーブル）を作成
        raceuma_df = pd.merge(raceuma_base_df, race_base_df, on="競走コード").rename(columns={"年月日": "年月日_x"})

        # 過去のレース全データを取得
        raceuma_prev_all_base_df = raceuma_result_df[["競走コード", "血統登録番号", "年月日", "馬番グループ", "距離増減", "タイム指数"]].copy()
        race_prev_all_base_df = race_result_df[["競走コード", "場コード", "距離", "ナイター", "季節", "非根幹", "距離グループ", "頭数グループ"]].copy()
        raceuma_prev_all_df = pd.merge(raceuma_prev_all_base_df, race_prev_all_base_df, on="競走コード").drop("競走コード", axis=1).rename(columns={"年月日": "年月日_y"})

        # 母数テーブルと過去走テーブルを血統登録番号で結合
        raceuma_prev_df = pd.set_prev_dfmerge(raceuma_df, raceuma_prev_all_df, on="血統登録番号")
        # 対象レースより前のレースのみに絞り込む
        raceuma_prev_df = raceuma_prev_df.query("年月日_x > 年月日_y")
        # 過去レースの結果を集計する
        self.grouped_raceuma_prev_df = self.tf.group_prev_raceuma_df(raceuma_prev_df, raceuma_base_df)


    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        result_race_df = self.ext.get_race_table_base()
        result_raceuma_df = self.ext.get_raceuma_table_base()
        result_haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.result_df = self._proc_result_df(result_race_df, result_raceuma_df, result_haraimodoshi_df)

    def _proc_result_df(self, result_race_df, result_raceuma_df, result_haraimodoshi_df):
        umaren_df = result_haraimodoshi_df[["競走コード", "馬連払戻金1"]].copy()
        result_df = pd.merge(result_race_df, result_raceuma_df, on="競走コード")
        result_df = pd.merge(result_df, umaren_df, on="競走コード")
        return result_df[["競走コード", "馬番", "月日", "確定着順", "複勝配当", "馬連払戻金1"]].copy()
