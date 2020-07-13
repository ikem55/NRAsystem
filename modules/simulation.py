from modules.extract import Extract

import sys
import pandas as pd
import numpy as np
import re
import itertools
import modules.util as mu

class Simulation(object):
    """
    馬券シミュレーションに関する処理をまとめたクラス
    """

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)

    def set_raceuma_df(self, raceuma_df):
        self.raceuma_df = raceuma_df

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Extract(start_date, end_date, mock_flag)
        return ext

    def _add_odds_list(self, df, fuku=False):
        df = df.assign(odds = "")
        for index, row in df.iterrows():
            odds_list = []
            for i in range(4):
                odds_list.append(row["全オッズ"][i::4])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [int(s)/ 10 for s in odds_list]
            if fuku:
                odds_list = odds_list[0::2]
            odds_list.insert(0,0)
            df["odds"][index] = odds_list
        return df

    def _add_odds_array(self, df, n, combi=True, fuku=False):
        df = df.assign(odds="")
        empty_check = ' ' * n
        for index, row in df.iterrows():
            odds_list = []
            tosu = row["登録頭数"] + 1
            for i in range(n):
                odds_list.append(row["全オッズ"][i::n])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [i for i in odds_list if i != empty_check]
            odds_list = [int(s)/ 10 for s in odds_list]
            if fuku:
                odds_list = odds_list[0::2]
            odds_array = np.zeros((tosu, tosu))
            idx = 0
            if combi:
                for element in itertools.combinations(range(1, tosu), 2):
                    odds_array[element[0], element[1]] = odds_list[idx]
                    odds_array[element[1], element[0]] = odds_list[idx]
                    idx = idx + 1
            else:
                for element in itertools.permutations(range(1, tosu), 2):
                    odds_array[element[0], element[1]] = odds_list[idx]
                    idx = idx + 1
            df["odds"][index] = odds_array
        return df

    def _add_odds_panel(self, df, n, combi=True):
        df = df.assign(odds="")
        empty_check = ' ' * n
        for index, row in df.iterrows():
            odds_list = []
            tosu = row["登録頭数"] + 1
            for i in range(n):
                odds_list.append(row["全オッズ"][i::n])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [i for i in odds_list if i != empty_check]
            odds_list = [int(s)/ 10 for s in odds_list]
            odds_array = np.zeros((tosu, tosu, tosu))
            idx = 0
            if combi:
                for element in itertools.combinations(range(1, tosu), 3):
                    odds_array[element[0], element[1], element[2]] = odds_list[idx]
                    odds_array[element[0], element[2], element[1]] = odds_list[idx]
                    odds_array[element[1], element[0], element[2]] = odds_list[idx]
                    odds_array[element[1], element[2], element[0]] = odds_list[idx]
                    odds_array[element[2], element[0], element[1]] = odds_list[idx]
                    odds_array[element[2], element[1], element[0]] = odds_list[idx]
                    idx = idx + 1
            else:
                for element in itertools.permutations(range(1, tosu), 3):
                    odds_array[element[0], element[1], element[2]] = odds_list[idx]
                    idx = idx + 1
            df["odds"][index] = odds_array
        return df

    def get_1tou_kaime(self, df1, odds_df):
        """ df1の買い目リストを作成, dfは競走コード,馬番のセット """
        # df1の馬番を横持に変換
        df1_gp = df1.groupby("競走コード")["馬番"].apply(list)
        merge_df = pd.merge(df1_gp, odds_df, on="競走コード")
        race_key_list = []
        umaban_list = []
        odds_list = []
        for index, row in merge_df.iterrows():
            uma1 = row["馬番"]
            for element in uma1:
                odds = row["odds"][element]
                race_key_list += [row["競走コード"]]
                umaban_list += [element]
                odds_list += [odds]
        kaime_df = pd.DataFrame(
            data={"競走コード": race_key_list, "馬番": umaban_list, "オッズ": odds_list},
            columns=["競走コード","馬番","オッズ"]
        )
        kaime_df = kaime_df[kaime_df["オッズ"] != 0]
        return kaime_df

    def get_2tou_kaime(self, df1, df2, odds_df, ren=True):
        """ df1とdf2の組み合わせの馬連の買い目リストを作成, dfは競走コード,馬番のセット """
        # df1の馬番を横持に変換
        df1_gp = df1.groupby("競走コード")["馬番"].apply(list)
        df2_gp = df2.groupby("競走コード")["馬番"].apply(list)
        merge_df = pd.merge(df1_gp, df2_gp, on="競走コード")
        merge_df = pd.merge(merge_df, odds_df, on="競走コード")
        race_key_list = []
        umaban_list = []
        check_umaban_list = []
        uma1_list = []
        uma2_list = []
        odds_list = []
        for index, row in merge_df.iterrows():
            uma1 = row["馬番_x"]
            uma2 = row["馬番_y"]
            for element in itertools.product(uma1, uma2):
                if element[0] != element[1]:
                    odds = row["odds"][element[0]][element[1]]
                    race_key_list += [row["競走コード"]]
                    if ren:
                        umaban_list += [sorted(element)] # 連系なのでソート
                        check_umaban_list += [str(sorted(element))]
                    else:
                        umaban_list += [element] # 単系の場合はソートしない
                        check_umaban_list += [str(element)]
                    uma1_list += [element[0]]
                    uma2_list += [element[1]]
                    odds_list += [odds]
        kaime_df = pd.DataFrame(
            data={"競走コード": race_key_list, "馬番": umaban_list, "オッズ": odds_list, "馬番1": uma1_list, "馬番2": uma2_list},
            columns=["競走コード","馬番","オッズ", "馬番1", "馬番2"]
        )
        kaime_df = kaime_df[kaime_df["オッズ"] != 0]
        kaime_df = kaime_df.drop_duplicates(subset=["競走コード", "チェック馬番"]).drop("チェック馬番", axis=1)
        return kaime_df

    def get_3tou_kaime(self, df1, df2, df3, odds_df, ren=True):
        """ df1とdf2.df3の組み合わせの馬連の買い目リストを作成, dfは競走コード,馬番のセット """
        #### 注意、馬１－馬２－馬３の組み合わせで馬１にある数字は馬２から除外されるようなので得点は小数点で計算する方がよさそう
        # df1の馬番を横持に変換
        df1_gp = df1.groupby("競走コード")["馬番"].apply(list)
        df2_gp = df2.groupby("競走コード")["馬番"].apply(list)
        df3_gp = df3.groupby("競走コード")["馬番"].apply(list)
        merge_df = pd.merge(df1_gp, df2_gp, on="競走コード")
        merge_df = pd.merge(merge_df, df3_gp, on="競走コード")
        merge_df = pd.merge(merge_df, odds_df, on="競走コード")
        race_key_list = []
        umaban_list = []
        check_umaban_list = []
        uma1_list = []
        uma2_list = []
        uma3_list = []
        odds_list = []
        for index, row in merge_df.iterrows():
            uma1 = row["馬番_x"]
            uma2 = row["馬番_y"]
            uma3 = row["馬番"]
            for element in itertools.product(uma1, uma2, uma3):
                if not(element[0] == element[1] or element[0] == element[2] or element[1] == element[2]):
                    odds = row["odds"][element[0]][element[1]][element[2]]
                    race_key_list += [row["競走コード"]]
                    if ren:
                        umaban_list += [sorted(element)] # 連系なのでソート
                        check_umaban_list += [str(sorted(element))]
                    else:
                        umaban_list += [element] # 単系の場合はソートしない
                        check_umaban_list += [str(element)]
                    uma1_list += [element[0]]
                    uma2_list += [element[1]]
                    uma3_list += [element[2]]
                    odds_list += [odds]
        kaime_df = pd.DataFrame(
            data={"競走コード": race_key_list, "馬番": umaban_list, "チェック馬番": check_umaban_list, "オッズ": odds_list, "馬番1": uma1_list, "馬番2": uma2_list, "馬番3": uma3_list},
            columns=["競走コード", "馬番", "チェック馬番", "オッズ", "馬番1", "馬番2", "馬番3"]
        )
        kaime_df = kaime_df[kaime_df["オッズ"] != 0]
        kaime_df = kaime_df.drop_duplicates(subset=["競走コード", "チェック馬番"]).drop("チェック馬番", axis=1)
        return kaime_df

    def check_result_kaime(self, kaime_df, result_df):
        """ 買い目DFと的中結果を返す """
        kaime_df["馬番"] = kaime_df["馬番"].apply(lambda x: ', '.join(map(str, x)))
        result_df["馬番"] = result_df["馬番"].apply(lambda x: ', '.join(map(str, x)))
        merge_df = pd.merge(kaime_df, result_df, on=["競走コード", "馬番"], how="left").fillna(0)
        return merge_df

    def calc_summary(self, df, cond_text):
        all_count = len(df)
        race_count = len(df["競走コード"].drop_duplicates())
        hit_df = df[df["払戻"] != 0]
        hit_count = len(hit_df)
        avg_return = round(hit_df["払戻"].mean(), 0)
        std_return = round(hit_df["払戻"].std(), 0)
        max_return = hit_df["払戻"].max()
        sum_return = hit_df["払戻"].sum()
        avg = round(df["払戻"].mean() , 1)
        hit_rate = round(hit_count / all_count * 100 , 1) if all_count !=0 else 0
        race_hit_rate = round(hit_count / race_count * 100 , 1) if race_count !=0 else 0
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"])
        return sr.fillna(0)

    def simulation_tansho(self, cond1):
        check_df = self.create_tansho_base_df(cond1)
        cond_text = cond1
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_tansho_base_df(self, cond1):
        self.sim_tansho()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        kaime_df = self.get_1tou_kaime(df1, self.tansho_df)
        check_df = pd.merge(kaime_df, self.result_tansho_df, on=["競走コード", "馬番"], how="left").fillna(0)
        return check_df

    def simulation_fukusho(self, cond1):
        check_df = self.create_fukusho_base_df(cond1)
        cond_text = cond1
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_fukusho_base_df(self, cond1):
        self.sim_fukusho()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        kaime_df = self.get_1tou_kaime(df1, self.fukusho_df)
        check_df = pd.merge(kaime_df, self.result_fukusho_df, on=["競走コード", "馬番"], how="left").fillna(0)
        return check_df

    def simulation_umaren(self, cond1, cond2):
        check_df = self.create_umaren_base_df(cond1, cond2)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_umaren_base_df(self, cond1, cond2):
        self.sim_umaren()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        df2 = self.raceuma_df.query(cond2)[["競走コード", "馬番"]]
        kaime_df = self.get_2tou_kaime(df1, df2, self.umaren_df)
        check_df = self.check_result_kaime(kaime_df, self.result_umaren_df)
        return check_df

    def simulation_wide(self, cond1, cond2):
        check_df = self.create_wide_base_df(cond1, cond2)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_wide_base_df(self, cond1, cond2):
        self.sim_wide()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        df2 = self.raceuma_df.query(cond2)[["競走コード", "馬番"]]
        kaime_df = self.get_2tou_kaime(df1, df2, self.wide_df)
        check_df = self.check_result_kaime(kaime_df, self.result_wide_df)
        return check_df

    def simulation_umatan(self, cond1, cond2):
        check_df = self.create_umatan_base_df(cond1, cond2)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_umatan_base_df(self, cond1, cond2):
        self.sim_umatan()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        df2 = self.raceuma_df.query(cond2)[["競走コード", "馬番"]]
        kaime_df = self.get_2tou_kaime(df1, df2, self.umatan_df, ren=False)
        check_df = self.check_result_kaime(kaime_df, self.result_umatan_df)
        return check_df


    def simulation_sanrenpuku(self, cond1, cond2, cond3):
        check_df = self.create_sanrenpuku_base_df(cond1, cond2, cond3)
        cond_text = "馬1." + cond1 + " AND 馬2." + cond2 + " AND 馬3." + cond3
        sr = self.calc_summary(check_df, cond_text)
        return sr

    def create_sanrenpuku_base_df(self, cond1, cond2, cond3):
        self.sim_sanrenpuku()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        df2 = self.raceuma_df.query(cond2)[["競走コード", "馬番"]]
        df3 = self.raceuma_df.query(cond3)[["競走コード", "馬番"]]
        kaime_df = self.get_3tou_kaime(df1, df2, df3, self.sanrenpuku_df)
        check_df = self.check_result_kaime(kaime_df, self.result_sanrenpuku_df)
        return check_df

    def sim_tansho(self):
        self._set_haraimodoshi_dict()
        self.result_tansho_df = self.dict_haraimodoshi["tansho_df"]
        self._set_tansho_df()

    def sim_fukusho(self):
        self._set_haraimodoshi_dict()
        self.result_fukusho_df = self.dict_haraimodoshi["fukusho_df"]
        self._set_fukusho_df()

    def sim_umaren(self):
        self._set_haraimodoshi_dict()
        self.result_umaren_df = self.dict_haraimodoshi["umaren_df"]
        self._set_umaren_df()

    def sim_wide(self):
        self._set_haraimodoshi_dict()
        self.result_wide_df = self.dict_haraimodoshi["wide_df"]
        self._set_wide_df()

    def sim_umatan(self):
        self._set_haraimodoshi_dict()
        self.result_umatan_df = self.dict_haraimodoshi["umatan_df"]
        self._set_umatan_df()

    def sim_sanrenpuku(self):
        self._set_haraimodoshi_dict()
        self.result_sanrenpuku_df = self.dict_haraimodoshi["sanrenpuku_df"]
        self._set_sanrenpuku_df()


    def _set_haraimodoshi_dict(self):
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.dict_haraimodoshi = mu.get_haraimodoshi_dict(haraimodoshi_df)

    def _set_tansho_df(self):
        """ tansho_df[umaban] """
        base_df = self.ext.get_tansho_table_base()
        odds_df = self._add_odds_list(base_df)
        self.tansho_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_fukusho_df(self):
        """ fukusho_df[umaban] """
        base_df = self.ext.get_fukusho_table_base()
        odds_df = self._add_odds_list(base_df, fuku=True)
        self.fukusho_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_umaren_df(self):
        """ umaren_df[umaban1][umaban2] """
        base_df = self.ext.get_umaren_table_base()
        odds_df = self._add_odds_array(base_df, 6)
        self.umaren_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_wide_df(self):
        """ wide_df[umaban1][umaban2] """
        base_df = self.ext.get_wide_table_base()
        odds_df = self._add_odds_array(base_df, 5, fuku=True)
        self.wide_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_umatan_df(self):
        """ umatan_df[umaban1][umaban2] """
        base_df = self.ext.get_umatan_table_base()
        odds_df = self._add_odds_array(base_df, 6, combi=False)
        self.umatan_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_sanrenpuku_df(self):
        """ sanrenpuku_df[umaban1][umaban2][umaban3] """
        base_df = self.ext.get_sanrenpuku_table_base()
        odds_df = self._add_odds_panel(base_df, 6)
        self.sanrenpuku_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

