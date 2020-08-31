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
    win_prob_list = [0, 0.4, 0.5, 0.6]
    jiku_prob_list = [0, 0.3, 0.4, 0.5]
    ana_prob_list = [0, 0.2, 0.3]
    #win_prob_list = [0, 0.5]
    #jiku_prob_list = [0, 0.4]
    #ana_prob_list = [0, 0.2]

    def __init__(self, start_date, end_date, mock_flag, raceuma_df):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)
        self.raceuma_df = raceuma_df.rename(columns={"RACE_KEY": "競走コード", "UMABAN": "馬番"})
        self._set_haraimodoshi_dict()

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
            df["odds"][index] = odds_list.copy()
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
            df["odds"][index] = odds_array.copy()
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
            df["odds"][index] = odds_array.copy()
        return df



    def calc_sim_df(self, type, cond, odds_cond):
        base_df = self._subproc_get_odds_base_df(type, cond)
        sim_df = base_df.query(odds_cond).copy()
        base_sr = self._calc_summary(sim_df, cond)
        return base_sr


    def calc_monthly_sim_df(self, type, cond):
        sim_df = self._subproc_get_odds_base_df(type, cond)
        summary_df = pd.DataFrame()
        ym_list = sim_df["年月"].drop_duplicates().tolist()
        for ym in ym_list:
            temp_ym_df = sim_df.query(f"年月 == '{ym}'")
            sim_sr = self._calc_summary(temp_ym_df, cond)
            sim_sr["年月"] = ym
            summary_df = summary_df.append(sim_sr, ignore_index=True)
        return summary_df

    def _subproc_get_odds_base_df(self, type, cond):
        result_df, raceuma_df = self._subproc_get_result_data(type)

        if type == "単勝" or type == "複勝":
            base_df = raceuma_df.query(cond).copy()
            #base_df = self.get_1tou_kaime(temp_raceuma1_df, odds_df)
            # base_df = pd.merge(temp_raceuma1_df, result_df, on=["競走コード", "馬番"], how="left").fillna(0)
        elif type == "馬連" or type == "ワイド":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            base_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df)
            #base_df = self.get_2tou_kaime(temp_raceuma1_df, temp_raceuma2_df, odds_df)
            #base_df = pd.merge(base_df, result_df, on=["競走コード", "馬番1", "馬番2"], how="left").fillna(0)
        elif type == "馬単":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            base_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df, ren=False)
            #base_df = self.get_2tou_kaime(temp_raceuma1_df, temp_raceuma2_df, odds_df, ren=False)
            #base_df = pd.merge(base_df, result_df, on=["競走コード",  "馬番1", "馬番2"], how="left").fillna(0)
        elif type == "三連複":
            temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
            temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
            temp_raceuma3_df = raceuma_df.query(cond[2]).copy()
            base_df = self._subproc_create_sanrenpuku_df(temp_raceuma1_df, temp_raceuma2_df, temp_raceuma3_df, result_df)
            #base_df = self.get_3tou_kaime(temp_raceuma1_df, temp_raceuma2_df, temp_raceuma3_df, odds_df)
            #base_df = pd.merge(base_df, result_df, on=["競走コード", "馬番1", "馬番2", "馬番3"], how="left").fillna(0)
        else:
            base_df = ""
        print(base_df.head())
        return base_df

    def _subproc_get_result_data(self, type):
        if type == "単勝":
            result_df = self.dict_haraimodoshi["tansho_df"]
            raceuma_df = pd.merge(self.raceuma_df, result_df, on=["競走コード", "馬番"], how="left").fillna(0)
        elif type == "複勝":
            result_df = self.dict_haraimodoshi["fukusho_df"]
            raceuma_df = pd.merge(self.raceuma_df, result_df, on=["競走コード", "馬番"], how="left").fillna(0)
        elif type == "馬連":
            result_df = self.dict_haraimodoshi["umaren_df"]
            result_df["馬番1"] = result_df["馬番"].apply(lambda x: x[0])
            result_df["馬番2"] = result_df["馬番"].apply(lambda x: x[1])
            raceuma_df = self.raceuma_df.copy()
        elif type == "馬単":
            result_df = self.dict_haraimodoshi["umatan_df"]
            result_df["馬番1"] = result_df["馬番"].apply(lambda x: x[0])
            result_df["馬番2"] = result_df["馬番"].apply(lambda x: x[1])
            raceuma_df = self.raceuma_df.copy()
        elif type == "ワイド":
            result_df = self.dict_haraimodoshi["wide_df"]
            result_df["馬番1"] = result_df["馬番"].apply(lambda x: x[0])
            result_df["馬番2"] = result_df["馬番"].apply(lambda x: x[1])
            raceuma_df = self.raceuma_df.copy()
        elif type == "三連複":
            result_df = self.dict_haraimodoshi["sanrenpuku_df"]
            result_df["馬番1"] = result_df["馬番"].apply(lambda x: x[0])
            result_df["馬番2"] = result_df["馬番"].apply(lambda x: x[1])
            result_df["馬番3"] = result_df["馬番"].apply(lambda x: x[2])
            raceuma_df = self.raceuma_df.copy()
        else:
            result_df = ""; raceuma_df = ""
        return result_df, raceuma_df

    def proc_fold_simulation(self, type):
        """ typeは単勝とか """
        ym_list = self.raceuma_df["年月"].drop_duplicates().tolist()
        # 対象券種毎の結果データを取得
        result_df, raceuma_df = self._subproc_get_result_data(type)

        # 的中件数が多く回収率が１０５％を超えている条件を抽出。的中件数の上位１０件を条件とする
        candidate_sim_df = pd.DataFrame()
        if type == "単勝" or type == "複勝":
            candidate_sim_df = self.proc_simulation_tanpuku(raceuma_df)
        elif type == "馬連" or type == "ワイド":
            candidate_sim_df = self.proc_simulation_umaren(raceuma_df, result_df)
        elif type == "馬単":
            candidate_sim_df = self.proc_simulation_umaren(raceuma_df, result_df, ren=False)
        elif type == "三連複":
            candidate_sim_df = self.proc_simulation_sanrenpuku(raceuma_df, result_df)

        if len(candidate_sim_df.index) == 0:
            print("対象なし")
            return pd.Series()
        else:
            target_sim_df = candidate_sim_df.sort_values("的中数", ascending=False).head(10)
            # 指定した条件の年月Foldを計算し、回収率１００％以上の件数が多い条件を採用。同数の場合は的中件数が多いものを採用
            print(target_sim_df)
            sim_df = pd.DataFrame()
            for index, row in target_sim_df.iterrows():
                cond = row["条件"]
                i = 0 #１００％超えカウント
                temp_raceuma1_df = ""; temp_raceuma2_df = ""; temp_raceuma3_df = "";
                if type == "単勝" or type == "複勝":
                    temp_raceuma1_df = raceuma_df.query(cond).copy()
                elif type == "馬連" or type == "馬単" or type == "ワイド":
                    temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
                    temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
                elif type == "三連複":
                    temp_raceuma1_df = raceuma_df.query(cond[0]).copy()
                    temp_raceuma2_df = raceuma_df.query(cond[1]).copy()
                    temp_raceuma3_df = raceuma_df.query(cond[2]).copy()
                for ym in ym_list:
                    temp_raceuma_df = ""
                    if type == "単勝" or type == "複勝":
                        temp_raceuma_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                    elif type == "馬連" or type == "ワイド":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_umaren_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, result_df)
                    elif type == "馬単":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_umaren_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, result_df, ren=False)
                    elif type == "三連複":
                        temp_ym_raceuma1_df = temp_raceuma1_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma2_df = temp_raceuma2_df.query(f"年月 == '{ym}'").copy()
                        temp_ym_raceuma3_df = temp_raceuma3_df.query(f"年月 == '{ym}'").copy()
                        temp_raceuma_df = self._subproc_create_sanrenpuku_df(temp_ym_raceuma1_df, temp_ym_raceuma2_df, temp_ym_raceuma3_df, result_df)
                    temp_sim_sr = self._calc_summary(temp_raceuma_df, cond)

                    if temp_sim_sr["回収率"] >= 100:
                        i = i + 1
                temp_final_sr = row
                temp_final_sr["条件数"] = i
                sim_df = sim_df.append(temp_final_sr, ignore_index=True)
            final_sim_df = sim_df.sort_values(["条件数", "的中数", "回収率"], ascending=False).reset_index()
            print(final_sim_df)
            return final_sim_df.iloc[0]


    def proc_simulation_tanpuku(self, raceuma_df):
        sim_df = pd.DataFrame()
        for rank in [1,2,3]:
            for win_prob in self.win_prob_list:
                for jiku_prob in self.jiku_prob_list:
                    for ana_prob in self.ana_prob_list:
                        cond = f"RANK <= {rank} and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                        temp_raceuma_df = raceuma_df.query(cond)
                        if len(temp_raceuma_df.index) != 0:
                            temp_sr = self._calc_summary(temp_raceuma_df, cond)
                            if temp_sr["回収率"] >= 105 and temp_sr["購入基準"] == "1":
                                sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df


    def proc_simulation_umaren(self, raceuma_df, result_df, ren=True):
        sim_df = pd.DataFrame()
        for rank1 in [1,2]:
            print(f"rank1: {rank1}")
            for win_prob in self.win_prob_list:
                for jiku_prob in self.jiku_prob_list:
                    for ana_prob in self.ana_prob_list:
                        cond1 = f"RANK <= {rank1} and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                        temp_raceuma1_df = raceuma_df.query(cond1).copy()
                        for rank2 in [3,4,5]:
                            if rank1 < rank2:
                                for win_prob2 in [0, 0.3, 0.5]:
                                    for jiku_prob2 in [0, 0.2, 0.4]:
                                        for ana_prob2 in [0, 0.2, 0.3]:
                                            cond2 = f"RANK <= {rank2} and win_prob >= {win_prob2} and jiku_prob >= {jiku_prob2} and ana_prob >= {ana_prob2}"
                                            temp_raceuma2_df = raceuma_df.query(cond2).copy()
                                            temp_raceuma_df = self._subproc_create_umaren_df(temp_raceuma1_df, temp_raceuma2_df, result_df, ren)
                                            if len(temp_raceuma_df.index) != 0:
                                                cond = [cond1, cond2]
                                                temp_sr = self._calc_summary(temp_raceuma_df, cond)
                                                if temp_sr["回収率"] >= 110 and temp_sr["購入基準"] == "1":
                                                    sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df

    def _subproc_create_umaren_df(self, raceuma1_df, raceuma2_df, result_df, ren=True):
        temp1_df = raceuma1_df[["競走コード", "馬番", "年月"]].copy()
        temp2_df = raceuma2_df[["競走コード", "馬番"]].copy()
        temp_df = pd.merge(temp1_df, temp2_df, on="競走コード")
        temp_df = temp_df.query("馬番_x != 馬番_y").copy()
        if len(temp_df.index) != 0:
            if ren:
                temp_df.loc[:, "馬番1"] = temp_df.apply(lambda x: x["馬番_x"] if x["馬番_x"] < x["馬番_y"] else x["馬番_y"], axis=1)
                temp_df.loc[:, "馬番2"] = temp_df.apply(lambda x: x["馬番_y"] if x["馬番_x"] < x["馬番_y"] else x["馬番_x"], axis=1)
                temp_df.loc[:, "馬番"] = temp_df.apply(lambda x: str(x["馬番_x"]) + ', ' + str(x["馬番_y"]) if x["馬番_x"] < x["馬番_y"] else str(x["馬番_y"]) + ', ' + str(x["馬番_x"]), axis=1)
                temp_df = temp_df.drop_duplicates(subset=["競走コード", "馬番"])
            else:
                temp_df.loc[:, "馬番1"] = temp_df["馬番_x"]
                temp_df.loc[:, "馬番2"] = temp_df["馬番_y"]
            kaime_df = temp_df[["競走コード", "馬番1", "馬番2", "年月"]].copy()
            merge_df = pd.merge(kaime_df, result_df, on=["競走コード", "馬番1", "馬番2"], how="left").fillna(0)
            return merge_df
        else:
            return pd.DataFrame()


    def proc_simulation_sanrenpuku(self, raceuma_df, result_df):
        sim_df = pd.DataFrame()
        for win_prob in self.win_prob_list:
            for jiku_prob in self.jiku_prob_list:
                for ana_prob in self.ana_prob_list:
                    cond1 = f"RANK == 1 and win_prob >= {win_prob} and jiku_prob >= {jiku_prob} and ana_prob >= {ana_prob}"
                    temp_raceuma1_df = raceuma_df.query(cond1).copy()
                    for rank2 in [2,3]:
                        for win_prob2 in [0, 0.3]:
                            for jiku_prob2 in [0, 0.2]:
                                for ana_prob2 in [0, 0.2]:
                                    cond2 = f"RANK <= {rank2} and win_prob >= {win_prob2} and jiku_prob >= {jiku_prob2} and ana_prob >= {ana_prob2}"
                                    temp_raceuma2_df = raceuma_df.query(cond2).copy()
                                    for rank3 in [5,6]:
                                        for win_prob3 in [0, 0.2]:
                                            for jiku_prob3 in [0, 0.1]:
                                                for ana_prob3 in [0, 0.1]:
                                                    cond3 = f"RANK <= {rank3} and win_prob >= {win_prob3} and jiku_prob >= {jiku_prob3} and ana_prob >= {ana_prob3}"
                                                    temp_raceuma3_df = raceuma_df.query(cond3).copy()
                                                    temp_raceuma_df = self._subproc_create_sanrenpuku_df(temp_raceuma1_df, temp_raceuma2_df, temp_raceuma3_df, result_df)
                                                    if len(temp_raceuma_df.index) != 0:
                                                        cond = [cond1, cond2, cond3]
                                                        temp_sr = self._calc_summary(temp_raceuma_df, cond)
                                                        if temp_sr["回収率"] >= 110 and temp_sr["購入基準"] == "1":
                                                            sim_df = sim_df.append(temp_sr, ignore_index=True)
        sim_df = sim_df.drop_duplicates(subset=["件数", "回収率", "払戻偏差"])
        return sim_df

    def _subproc_create_sanrenpuku_df(self, raceuma1_df, raceuma2_df, raceuma3_df, result_df):
        temp1_df = raceuma1_df[["競走コード", "馬番", "年月"]].copy()
        temp2_df = raceuma2_df[["競走コード", "馬番"]].copy()
        temp3_df = raceuma3_df[["競走コード", "馬番"]].copy()
        temp_df = pd.merge(temp1_df, temp2_df, on="競走コード")
        temp_df = pd.merge(temp_df, temp3_df, on="競走コード")
        temp_df = temp_df.query("馬番_x != 馬番_y and 馬番_x != 馬番 and 馬番_y != 馬番").copy()
        if len(temp_df.index) != 0:
            temp_df.loc[:, "連番"] = temp_df.apply(lambda x: sorted(list([x["馬番_x"], x["馬番_y"], x["馬番"]])), axis=1)
            temp_df.loc[:, "馬番1"] = temp_df["連番"].apply(lambda x: x[0])
            temp_df.loc[:, "馬番2"] = temp_df["連番"].apply(lambda x: x[1])
            temp_df.loc[:, "馬番3"] = temp_df["連番"].apply(lambda x: x[2])
            temp_df.loc[:, "馬番"] = temp_df["連番"].apply(lambda x: ', '.join(map(str, x)))
            temp_df = temp_df.drop_duplicates(subset=["競走コード", "馬番"])
            kaime_df = temp_df[["競走コード", "馬番1", "馬番2", "馬番3", "年月"]].copy()
            merge_df = pd.merge(kaime_df, result_df, on=["競走コード", "馬番1", "馬番2", "馬番3", ], how="left").fillna(0)
            return merge_df
        else:
            return pd.DataFrame()



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
            data={"競走コード": race_key_list, "馬番": umaban_list,  "チェック馬番": check_umaban_list, "オッズ": odds_list, "馬番1": uma1_list, "馬番2": uma2_list},
            columns=["競走コード","馬番","チェック馬番", "オッズ", "馬番1", "馬番2"]
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

    def get_odds_df(self, type):
        if type == "単勝":
            self._set_tansho_df()
            return self.tansho_df
        elif type == "複勝":
            self._set_fukusho_df()
            return self.fukusho_df
        elif type == "馬連":
            self._set_umaren_df()
            return self.umaren_df
        elif type == "馬単":
            self._set_umatan_df()
            return self.umatan_df
        elif type == "ワイド":
            self._set_wide_df()
            return self.wide_df
        elif type == "三連複":
            self._set_sanrenpuku_df()
            return self.sanrenpuku_df


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


    def _calc_summary(self, df, cond_text):
        all_count = len(df)
        race_count = len(df["競走コード"].drop_duplicates())
        hit_df = df[df["払戻"] != 0]
        hit_count = len(hit_df)
        race_hit_count = len(hit_df["競走コード"].drop_duplicates())
        avg_return = round(hit_df["払戻"].mean(), 0)
        std_return = round(hit_df["払戻"].std(), 0)
        max_return = hit_df["払戻"].max()
        sum_return = hit_df["払戻"].sum()
        avg = round(df["払戻"].mean() , 1)
        hit_rate = round(hit_count / all_count * 100 , 1) if all_count !=0 else 0
        race_hit_rate = round(race_hit_count / race_count * 100 , 1) if race_count !=0 else 0
        vote_check = "1" if sum_return - max_return > all_count * 100 else "0"
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return, vote_check]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額", "購入基準"])
        return sr.fillna(0)