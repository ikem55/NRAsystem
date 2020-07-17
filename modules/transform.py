import os
import modules.util as mu

import pandas as pd
import numpy as np
import os
import math
import sys
from factor_analyzer import FactorAnalyzer


class Transform(object):
    """
    データ変換に関する共通処理を定義する
    辞書データの格納等の処理があるため、基本的にはインスタンス化して実行するが、不要な場合はクラスメソッドでの実行を可とする。
    辞書データの作成は、ない場合は作成、ある場合は読み込み、を基本方針とする(learning_modeによる判別はしない）
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date


    def del_choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列はデータ区分、主催者コード、競走コード、月日、距離、場コード、頭数、予想勝ち指数、予想決着指数, 競走種別コード

        :param dataframe race_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_race_df = race_df[
            ['データ区分', '主催者コード', '競走コード', '月日', '距離', '場コード', '頭数', '天候コード', '馬場状態コード', '競走種別コード', 'ペース', '後３ハロン']]
        return temp_race_df


    ##################### race_df ###############################
    def encode_race_df(self, race_df):
        """  列をエンコードする処理（ラベルエンコーディング、onehotエンコーディング等）"""
        race_df.loc[:, "競走条件コード"] = race_df["競走条件コード"].apply(lambda x: mu.convert_kyoso_joken_code(x))
        return race_df

    def normalize_race_df(self, race_df):
        return race_df

    def standardize_race_df(self, race_df):
        return race_df

    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        race_df.loc[:, '月'] = race_df['月日'].apply(lambda x: x.month)
        race_df.loc[:, 'ナイター'] = race_df['発走時刻'].apply(lambda x: 1 if x.hour >= 17 else 0)
        race_df.loc[:, '季節'] = (race_df['月日'].apply(lambda x: x.month) - 1) // 3
        race_df['季節'].astype('str')
        race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        race_df.loc[:, "頭数グループ"] = race_df["頭数"] // 5
        race_df.loc[:, "コース"] = race_df["場コード"].astype(str) + race_df["トラックコード"].astype(str)
        race_df.loc[:, 'cos_day'] = race_df['月日'].dt.dayofyear.apply(
            lambda x: np.cos(math.radians(90 - (x / 365) * 360)))
        race_df.loc[:, 'sin_day'] = race_df['月日'].dt.dayofyear.apply(
            lambda x: np.sin(math.radians(90 - (x / 365) * 360)))
        return race_df

    def choose_race_df_columns(self, race_df):
        race_df = race_df[['競走コード', '月日', '距離', '競走番号', '場コード', '場名', '主催者コード',
       '競走種別コード', '発走時刻', '頭数', 'トラックコード', '予想勝ち指数',
       '初出走頭数', '混合', '予想決着指数', '登録頭数', '回次', '日次', '月', 'ナイター', '季節', '非根幹',
       '距離グループ', '頭数グループ', 'コース']].copy()
        race_df = race_df.astype({'場コード': int, '競走種別コード': int, 'トラックコード': int, 'コース': int,
                                  '主催者コード': int})
        return race_df

    ##################### raceuma_df ###############################
    def encode_raceuma_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        raceuma_df = self.choose_upper_n_count(raceuma_df, "騎手名", 100, dict_folder)
        raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder)
        raceuma_df = self.choose_upper_n_count(raceuma_df, "調教師名", 100, dict_folder)
        raceuma_df.loc[:, '調教師名'] = mu.label_encoding(raceuma_df['調教師名'], '調教師名', dict_folder)
        raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder)
        raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder)
        raceuma_df.loc[:, '予想展開'] = raceuma_df["予想展開"].astype(str)
        return raceuma_df

    def normalize_raceuma_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        grouped_df = raceuma_df[['競走コード', '負担重量', '予想タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3']].groupby('競走コード').agg(
            ['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', '負担重量_mean', '負担重量_std', '予想タイム指数_mean', '予想タイム指数_std', 'デフォルト得点_mean',
                              'デフォルト得点_std', '得点V1_mean', '得点V1_std', '得点V2_mean', '得点V2_std', '得点V3_mean', '得点V3_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['負担重量偏差'] = (merged_df['負担重量'] - merged_df['負担重量_mean']) / merged_df['負担重量_std'] * 10 + 50
        merged_df['予想タイム指数偏差'] = (merged_df['予想タイム指数'] - merged_df['予想タイム指数_mean']) / merged_df['予想タイム指数_std'] * 10 + 50
        merged_df['デフォルト得点偏差'] = (merged_df['デフォルト得点'] - merged_df['デフォルト得点_mean']) / merged_df['デフォルト得点_std'] * 10 + 50
        merged_df['得点V1偏差'] = (merged_df['得点V1'] - merged_df['得点V1_mean']) / merged_df['得点V1_std'] * 10 + 50
        merged_df['得点V2偏差'] = (merged_df['得点V2'] - merged_df['得点V2_mean']) / merged_df['得点V2_std'] * 10 + 50
        merged_df['得点V3偏差'] = (merged_df['得点V3'] - merged_df['得点V3_mean']) / merged_df['得点V3_std'] * 10 + 50
        merged_df.drop(
            ['負担重量_mean', '負担重量_std', '予想タイム指数_mean', '予想タイム指数_std', 'デフォルト得点_mean', 'デフォルト得点_std', '得点V1_mean',
             '得点V1_std',
             '得点V2_mean', '得点V2_std', '得点V3_mean', '得点V3_std', '負担重量', '予想タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3'],
            axis=1, inplace=True)
        raceuma_df = merged_df.rename(columns={'負担重量偏差': '負担重量', '予想タイム指数偏差': '予想タイム指数',
                                               'デフォルト得点偏差': 'デフォルト得点', '得点V1偏差': '得点V1', '得点V2偏差': '得点V2',
                                               '得点V3偏差': '得点V3'}).copy()
        raceuma_df.fillna({'負担重量': 50, '予想タイム指数': 50, 'デフォルト得点': 50, '得点V1': 50, '得点V2': 50, '得点V3': 50}, inplace=True)
        return raceuma_df.copy()

    def standardize_raceuma_df(self, raceuma_df):
        """ 数値データを整備する。無限大(inf）をnanに置き換える

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, '予想タイム指数'] = raceuma_df['予想タイム指数'].replace([np.inf, -np.inf], np.nan)
        return temp_raceuma_df.copy()

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["馬番"] // 4
        temp_raceuma_df.loc[:, "予想タイム指数順位"] = raceuma_df["予想タイム指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養週数"] = raceuma_df["休養週数"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養後出走回数"] = raceuma_df["休養後出走回数"].apply(lambda x: 5 if x >= 5 else x)
        temp_raceuma_df.loc[:, "予想人気"] = raceuma_df["予想人気"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "先行指数順位"] = raceuma_df["先行指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "キャリア"] = raceuma_df["キャリア"].apply(lambda x: 10 if x >= 10 else x)
        temp_raceuma_df.loc[:, "馬齢"] = raceuma_df["馬齢"].apply(lambda x: 7 if x >= 7 else x)
        temp_raceuma_df.loc[:, "距離増減"] = raceuma_df["距離増減"] // 200
        return temp_raceuma_df

    def choose_raceuma_df_columns(self, raceuma_df):
        raceuma_df = raceuma_df[['競走コード', '馬番', '枠番', '血統登録番号', '性別コード', '年月日', '予想タイム指数順位',
       '近走競走コード1', '近走馬番1', '近走競走コード2', '近走馬番2', '近走競走コード3', '近走馬番3',
       '近走競走コード4', '近走馬番4', '近走競走コード5', '近走馬番5', '休養週数', '休養後出走回数', '予想オッズ',
       '予想人気', '血統距離評価', '血統トラック評価', '血統成長力評価', '血統総合評価', '血統距離評価B',
       '血統トラック評価B', '血統成長力評価B', '血統総合評価B', '先行指数', '先行指数順位', '予想展開', 'クラス変動',
       '騎手コード', '騎手所属場コード', '見習区分', '騎手名', 'テン乗り', '騎手評価', '調教師評価', '枠順評価',
       '脚質評価', 'キャリア', '馬齢', '調教師コード', '調教師所属場コード', '調教師名', '距離増減', '前走着順',
       '前走人気', '前走着差', '前走トラック種別コード', '前走馬体重', '前走頭数', 'タイム指数上昇係数',
       'タイム指数回帰推定値', 'タイム指数回帰標準偏差', '所属', '転厩', '斤量比', '前走休養週数', '騎手ランキング',
       '調教師ランキング', '得点V1順位', '得点V2順位', 'デフォルト得点順位', '得点V3順位', '負担重量',
       '予想タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3', '馬番グループ']].copy()
        raceuma_df = raceuma_df.astype({'性別コード': int, '予想展開': int, '騎手コード': int, '騎手所属場コード': int,
                                        '見習区分': int, '調教師コード': int, '調教師所属場コード': int, '前走トラック種別コード': int})
        return raceuma_df

    ##################### horse_df ###############################
    def encode_horse_df(self, race_df):
        """  列をエンコードする処理（ラベルエンコーディング、onehotエンコーディング等）"""
        return race_df

    def normalize_horse_df(self, race_df):
        return race_df

    def standardize_horse_df(self, race_df):
        return race_df

    def create_feature_horse_df(self, race_df):
        return race_df

    def choose_horse_df_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        horse_df = horse_df[['血統登録番号', '馬名', 'JRA施設在厩フラグ', '馬記号コード',
       '品種コード', '毛色コード', '繁殖登録番号1', '繁殖馬名1', '繁殖登録番号3',
       '繁殖馬名3', '繁殖登録番号5', '繁殖馬名5', '東西所属コード',
       '生産者コード', '生産者名', '産地名', '馬主コード', '馬主名']].copy()
        horse_df = horse_df.astype({'JRA施設在厩フラグ': int, '馬記号コード': int, '品種コード': int, '毛色コード': int,
                                    '繁殖登録番号1': int, '繁殖登録番号3': int, '繁殖登録番号5': int, '東西所属コード': int,
                                    '生産者コード': int, '馬主コード': int})
        return horse_df


    ##################### race_result_df ###############################
    def encode_race_result_df(self, race_df):
        race_df = self.encode_race_df(race_df)
        race_df.loc[:, "ペース"] = race_df["ペース"].apply(lambda x: mu.convert_pace(x))
        return race_df

    def normalize_race_result_df(self, race_df):
        race_df = self.normalize_race_df(race_df)
        return race_df

    def standardize_race_result_df(self, race_df):
        race_df = self.standardize_race_df(race_df)
        return race_df

    def create_feature_race_result_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        race_df = self.create_feature_race_df(race_df)
        race_df.loc[:, "上り係数"] = race_df.apply(
            lambda x: 1 if (x["後３ハロン"] == 0 or x["一位入線タイム"] == 0) else (x["後３ハロン"] / 600) / (x["一位入線タイム"] / x["距離"]), axis=1)
        return race_df

    def choose_race_result_df_columns(self, race_df):
        race_df = race_df[['データ作成年月日', '競走コード', '距離', '主催者コード', '競走番号', '場コード', '場名','頭数',
       '競走種別コード', '競走条件コード', 'トラックコード', 'トラック種別コード',
       '天候コード', '前３ハロン', '前４ハロン', '後３ハロン', '後４ハロン', '馬場状態コード', '前半タイム',
       'ペース', '初出走頭数', '混合', '人気馬支持率1',
       '人気馬支持率2', '人気馬支持率3',
       '予想決着指数', '波乱度', 'タイム指数誤差',
       '月', 'ナイター', '季節', '非根幹',
       '距離グループ', '頭数グループ', 'コース', 'cos_day', 'sin_day', '上り係数']].copy()
        race_df = race_df.astype({'主催者コード': int, '場コード': int,
                                  '競走種別コード': int, 'トラックコード': int, 'トラック種別コード': int,
                                  '天候コード': int, '馬場状態コード': int, '混合': int, 'コース': int, '波乱度': int})
        return race_df


    ######## Race単位の結果データ作成
    def create_feature_race_result_winner_df(self, race_df, race_winner_df):
        """  race_ddfのデータから特徴量を作成して列を追加する。月日→月、距離→非根幹、距離グループを作成

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df = pd.merge(temp_race_df, race_winner_df, on="競走コード")
        temp_race_df.loc[:, "逃げ勝ち"] = temp_race_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        temp_race_df.loc[:, "内勝ち"] = temp_race_df["枠番"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
        temp_race_df.loc[:, "外勝ち"] = temp_race_df["枠番"].apply(lambda x: 1 if x in (6, 7, 8) else 0)
        temp_race_df.loc[:, "短縮勝ち"] = temp_race_df["距離増減"].apply(lambda x: 1 if x < 0 else 0)
        temp_race_df.loc[:, "延長勝ち"] = temp_race_df["距離増減"].apply(lambda x: 1 if x > 0 else 0)
        temp_race_df.loc[:, "人気勝ち"] = temp_race_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        race_df = pd.merge(race_df,
                            temp_race_df[["競走コード", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]],
                            on="競走コード")
        return race_df

    ####### 過去走用に不要な列を削除
    def drop_race_result_df_columns(self, race_result_df):
        race_result_df = race_result_df.drop(["データ作成年月日", "競走番号", "場名", '競走種別コード', '競走条件コード', 'トラックコード', 'トラック種別コード',
                                              '天候コード', '前３ハロン', '前４ハロン', '後３ハロン', '後４ハロン', '馬場状態コード', '前半タイム'], axis=1)
        return race_result_df

    ##################### raceuma_result_df ###############################
    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        raceuma_df = self.encode_raceuma_df(raceuma_df, dict_folder)
        raceuma_df.loc[:, '展開脚質'] = raceuma_df['展開コード'].astype(str).str[:1].astype(int)
        raceuma_df.loc[:, '展開脚色'] = raceuma_df['展開コード'].astype(str).str[-1:].astype(int)
        raceuma_df = raceuma_df.drop('展開コード', axis=1)
        return raceuma_df

    def normalize_raceuma_result_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        raceuma_df = self.normalize_raceuma_df(raceuma_df)
        grouped_df = raceuma_df[['競走コード', 'タイム指数', '上がりタイム']].groupby('競走コード').agg(
            ['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', 'タイム指数_mean', 'タイム指数_std', '上がりタイム_mean', '上がりタイム_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['タイム指数偏差'] = (merged_df['タイム指数'] - merged_df['タイム指数_mean']) / merged_df['タイム指数_std'] * 10 + 50
        merged_df['上がりタイム偏差'] = merged_df.apply(lambda x: 50 if x['上がりタイム_std'] == 0 else (x['上がりタイム_mean'] - x['上がりタイム'])/ x['上がりタイム_std'] * 10 + 50 , axis=1)
        merged_df['上がりタイム順位'] = raceuma_df.groupby("競走コード")["上がりタイム"].rank(method='min')
        merged_df['上がりタイム順位'] = merged_df.apply(lambda x: np.nan if x["上がりタイム"] == 0 else x["上がりタイム順位"], axis=1)
        merged_df.drop(
            ['タイム指数_mean', 'タイム指数_std', '上がりタイム_mean', '上がりタイム_std', 'タイム指数'],
            axis=1, inplace=True)
        raceuma_df = merged_df.rename(columns={'タイム指数偏差': 'タイム指数', '上がりタイム偏差': '上がり偏差', '上がりタイム順位': '上がり順位'}).copy()
        raceuma_df.replace([np.inf, -np.inf], np.nan)
        raceuma_df.fillna({'タイム指数': 50, '上がり偏差': 50}, inplace=True)
        return raceuma_df.copy()


    def standardize_raceuma_result_df(self, raceuma_df):
        raceuma_df = self.standardize_raceuma_df(raceuma_df)
        raceuma_df.loc[:, 'タイム指数'] = raceuma_df['タイム指数'].replace([np.inf, -np.inf], np.nan)
        return raceuma_df

    def create_feature_raceuma_result_df(self, raceuma_df):
        raceuma_df = self.create_feature_raceuma_df(raceuma_df)
        raceuma_df.loc[:, "勝ち"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "１番人気"] = raceuma_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "３角先頭"] = raceuma_df["コーナー順位3"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "４角先頭"] = raceuma_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "上がり最速"] = raceuma_df["上がり順位"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "休み明け"] = raceuma_df["休養週数"].apply(lambda x: 1 if x >= 10 else 0)
        raceuma_df.loc[:, "連闘"] = raceuma_df["休養週数"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "大差負け"] = raceuma_df["着差"].apply(lambda x: 1 if x >= 20 else 0)
        raceuma_df.loc[:, "凡走"] = raceuma_df.apply(lambda x: 1 if x["確定着順"] - x["単勝人気"] > 5 else 0, axis=1)
        raceuma_df.loc[:, "好走"] = raceuma_df["確定着順"].apply(lambda x: 1 if x <= 3 else 0)
        raceuma_df.loc[:, "激走"] = raceuma_df.apply(lambda x: 1 if x["単勝人気"] - x["確定着順"] > 5 else 0, axis=1)
        raceuma_df.loc[:, "逃げそびれ"] = raceuma_df.apply(lambda x: 1 if x["予想展開"] == "1" and - x["コーナー順位4"] > 3 else 0, axis=1)
        return raceuma_df

    def choose_raceuma_result_df_columns(self, raceuma_df):
        raceuma_df = raceuma_df[['競走コード', '馬番', '枠番', '血統登録番号', 'タイム', '年月日',
                                 '着差',
                                 '休養週数', '休養後出走回数', '単勝配当', '複勝配当', '単勝オッズ', '単勝人気', '単勝支持率', '複勝オッズ1', '複勝オッズ2',
                                 '予想オッズ', '予想人気',
                                 '投票直前単勝オッズ', '投票直前複勝オッズ','上がりタイム',
                                 '先行率', 'ペース偏差値', 'クラス変動', '騎手コード', '騎手所属場コード',
                                 '見習区分', '騎手名', 'テン乗り', '馬齢', '調教師所属場コード',
                                 '調教師名', '馬体重', '馬体重増減', '異常区分コード', '確定着順', 'コーナー順位1', 'コーナー順位2', 'コーナー順位3', 'コーナー順位4',
                                 '距離増減', '所属', '転厩', '斤量比', '前走休養週数', '騎手ランキング', '調教師ランキング', '馬単合成オッズ',
                                 '展開脚質', '展開脚色', '負担重量', '予想タイム指数', 'デフォルト得点', '得点V1',
                                 '得点V2', '得点V3', 'タイム指数', '上がり偏差', '上がり順位', '馬番グループ', '勝ち', '１番人気', '３角先頭', '４角先頭', '上がり最速',
                                 '休み明け', '連闘', '大差負け', '凡走', '好走', '激走', '逃げそびれ']].copy()
        raceuma_df = raceuma_df.astype({'騎手コード': int, '騎手所属場コード': int, '見習区分': int, 'テン乗り': int, '調教師所属場コード': int})
        return raceuma_df

    ######## RaceUma単位の結果データ作成
    def create_feature_raceuma_result_race_df(self, race_df, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_merge_df = pd.merge(race_df, raceuma_df, on="競走コード")
        print("create_feature_raceuma_result_df: temp_merge_df", temp_merge_df.shape)
        temp_merge_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
        temp_merge_df.loc[:, "平均タイム"] = temp_merge_df["タイム"] / temp_merge_df["距離"] * 200
        temp_merge_df.loc[:, "同場騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_merge_df.loc[:, "同所属場"] = (temp_merge_df["調教師所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_merge_df.loc[:, "同所属騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["調教師所属場コード"]).astype(int)
        raceuma_df = pd.merge(raceuma_df,
                              temp_merge_df[["競走コード", "馬番", "追込率", "平均タイム",  "同場騎手", "同所属場", "同所属騎手"]],
                            on=["競走コード", "馬番"])

        return raceuma_df

    ######## 過去走用に不要な列を削除
    def drop_raceuma_result_df_columns(self, raceuma_result_df):
        raceuma_result_df = raceuma_result_df.drop(["単勝配当", "複勝配当", "予想オッズ", "予想人気", "投票直前単勝オッズ", "投票直前複勝オッズ", "上がりタイム",
                                                    "騎手名", "調教師名", "所属", "転厩", "異常区分コード"], axis=1)
        return raceuma_result_df

    def drop_prev_raceuma_result_df_columns(self, raceuma_result_df):
        raceuma_result_df = raceuma_result_df.drop(['年月日', '血統登録番号'], axis=1)
        return raceuma_result_df

    ########## RaceUma単位で過去走を集計したもの
    def group_prev_raceuma_df(self, raceuma_prev_df, raceuma_base_df):
        """
        | set_all_prev_raceuma_dfで生成された過去レコードに対して、条件毎にタイム指数の最大値と平均値を取得してdataframeとして返す。normalize_group_prev_raceuma_dfで最終的に偏差値化する。
        | 条件は、同場、同距離、同根幹、同距離グループ、同馬番グループ

        :param dataframe raceuma_prev_df: dataframe（過去走のdataframe)
        :param dataframe raceuma_base_df: dataframe（軸となるdataframe)
        :return: dataframe
        """
        # ベースとなるデータフレームを作成する
        raceuma_base_df = raceuma_base_df[["競走コード", "馬番"]]
        # 集計対象の条件のみに絞り込んだデータを作成し、平均値と最大値を計算する
        same_ba_df = raceuma_prev_df.query("場コード_x == 場コード_y").groupby(["競走コード", "馬番"])["タイム指数"].max().reset_index().rename(columns={'タイム指数': '同場コード_max'})
        # 集計したデータをベースのデータフレームに結合追加する
        raceuma_base_df = pd.merge(raceuma_base_df, same_ba_df, on=["競走コード", "馬番"], how='left')
        group_columns = ["競走コード", "同場コード_max"]
        group_calc_columns = ["競走コード", "同場コード_max_mean", "同場コード_max_std"]
        for type in ["距離", "ナイター", "季節", "非根幹", "距離グループ", "頭数グループ", "馬番グループ", "距離増減"]:
            temp_df = raceuma_prev_df.query(f"{type}_x == {type}_y").groupby(["競走コード", "馬番"])["タイム指数"].max().reset_index().rename(columns={"タイム指数": f"同{type}_max"})
            raceuma_base_df = pd.merge(raceuma_base_df, temp_df, on=["競走コード", "馬番"], how='left')
            group_columns.append(f"同{type}_max")
            group_calc_columns.append(f"同{type}_max_mean")
            group_calc_columns.append(f"同{type}_max_std")

        # 集計値を標準化する
        # 偏差値計算するための平均値と偏差を計算する
        grouped_df = raceuma_base_df[group_columns].groupby("競走コード").agg(["mean", "std"]).reset_index()
        # 計算するために列名を変更する
        grouped_df.columns = group_calc_columns
        # 偏差値計算するためのデータフレームを作成する
        merged_df = pd.merge(raceuma_base_df, grouped_df, on='競走コード')
        # ベースとなるデータフレームを作成する
        df = merged_df[["競走コード", "馬番"]].copy()
        # 各偏差値をベースとなるデータフレームに追加していく
        for type in ["場コード", "距離", "ナイター", "季節", "非根幹", "距離グループ", "頭数グループ", "馬番グループ", "距離増減"]:
            df.loc[:, f'同{type}_max'] = (merged_df[f'同{type}_max'] - merged_df[f'同{type}_max_mean']) / merged_df[f'同{type}_max_std'] * 10 + 50
        return df



    def choose_raceuma_result_column(self, raceuma_df):
        """  レース馬データから必要な列に絞り込む。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df[
            ['データ区分', '競走コード', '馬番', '年月日', '血統登録番号', 'タイム指数', '単勝オッズ', '単勝人気', '確定着順', '着差', '休養週数', '先行率', 'タイム',
             'ペース偏差値', '展開コード', 'クラス変動', '騎手所属場コード', '騎手名', 'テン乗り', '負担重量', '馬体重', '馬体重増減', 'コーナー順位4', '距離増減', '所属',
             '調教師所属場コード',
             '転厩', '斤量比', '騎手ランキング', '調教師ランキング', 'デフォルト得点', '得点V1', '得点V2', '得点V3']].copy()
        return temp_raceuma_df




    def _convert_ninki_group(self, ninki):
        if ninki == 1:
            return 1
        elif ninki in (2, 3):
            return 2
        elif ninki in (4, 5, 6):
            return 3
        else:
            return 4


    def choose_horse_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df[['血統登録番号', '繁殖登録番号1', '繁殖登録番号5', '東西所属コード', '生産者コード', '馬主コード']]
        return temp_horse_df

    def normalize_prev_merged_df(self, merged_df):
        """ 前走レース馬データ用に値を平準化した値に置き換える。頭数で割る or 逆数化。対象は単勝人気、確定着順、コーナー順位４、騎手ランキング、調教師ランキング

        :param dataframe merged_df:
        :return: dataframe
        """
        temp_merged_df = merged_df.copy()
        temp_merged_df.loc[:, '単勝人気'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['単勝人気'] / x['頭数'], axis=1)
        temp_merged_df.loc[:, '確定着順'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['確定着順'] / x['頭数'], axis=1)
        temp_merged_df.loc[:, 'コーナー順位4'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['コーナー順位4'] / x['頭数'],
                                                           axis=1)
        # temp_merged_df.loc[:, '騎手ランキング'] = merged_df.apply(lambda x: np.nan if x['騎手ランキング'] == 0 else 1 / x['騎手ランキング'], axis=1)
        # temp_merged_df.loc[:, '調教師ランキング'] = merged_df.apply(lambda x: np.nan if x['調教師ランキング'] == 0 else 1 / x['調教師ランキング'], axis=1)
        return temp_merged_df

    def choose_upper_n_count(self, df, column_name, n, dict_folder):
        """ 指定したカラム名の上位N出現以外をその他にまとめる

        :param df:
        :param column_name:
        :param n:
        :return: df
        """
        dict_name = "choose_upper_" + str(n) + "_" + column_name
        file_name = dict_folder + dict_name + ".pkl"
        if os.path.exists(file_name):
            temp_df = mu.load_dict(dict_name, dict_folder)
        else:
            temp_df = df[column_name].value_counts().iloc[:n].index
            mu.save_dict(temp_df, dict_name, dict_folder)
        df.loc[:, column_name] = df[column_name].apply(lambda x: x if x in temp_df else 'その他')
        return df