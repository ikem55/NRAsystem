from modules.extract import Extract
import modules.util as mu
import my_config as mc

import pandas as pd
import dropbox
from datetime import datetime as dt


class Report(object):
    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.ext = Extract(start_date, end_date, mock_flag)
        self._set_predata()
        self.dbx = dropbox.Dropbox(mc.DROPBOX_KEY)

    def _set_predata(self):
        self._set__bet_df()
        self._set_haraimodoshi_dict()
        self._set_raceuma_df()
        self._check_result_data()

    def _check_result_data(self):
        if len(self.bet_df[self.bet_df["日付"] == self.end_date]) != 0:
            self.check_flag = True
        else:
            self.check_flag = False

    def _set__bet_df(self):
        base_bet_df = self.ext.get_bet_table_base()
        bet_df = base_bet_df[["競走コード", "式別", "日付", "結果", "金額"]].copy()
        bet_df.loc[:, "結果"] = bet_df["結果"] * bet_df["金額"] / 100
        self.bet_df = bet_df

    def _set_haraimodoshi_dict(self):
        base_haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        end_date = self.end_date
        self.todays_haraimodoshi_dict = mu.get_haraimodoshi_dict(base_haraimodoshi_df.query("データ作成年月日 == @end_date"))
        self.haraimodoshi_dict = mu.get_haraimodoshi_dict(base_haraimodoshi_df)

    def _set_raceuma_df(self):
        base_race_df = self.ext.get_race_table_base()
        base_raceuma_df = self.ext.get_raceuma_table_base()
        self.race_df = base_race_df[["競走コード", "データ区分", "月日", "距離", "競走番号", "場名", "発走時刻", "投票フラグ"]]
        raceuma_df = base_raceuma_df.drop("データ区分", axis=1)#[["競走コード", "馬番", "年月日", "馬券評価順位", "得点", "単勝オッズ", "単勝人気", "確定着順", "単勝配当", "複勝配当", "デフォルト得点順位"]].copy()
        raceuma_df.loc[:, "ck1"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "ck2"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 2 else 0)
        raceuma_df.loc[:, "ck3"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 3 else 0)
        raceuma_df.loc[:, "ckg"] = raceuma_df["確定着順"].apply(lambda x: 1 if x > 3 else 0)
        self.raceuma_df = pd.merge(raceuma_df, self.race_df[["競走コード", "場名", "距離", "データ区分"]], on="競走コード")


    def export_bet_df(self):
        bet_df = self.bet_df
        bet_df.loc[:, "月日"] = bet_df["日付"].apply(lambda x: str(x.year) + str(x.month))
        month_list = bet_df["月日"].drop_duplicates().tolist()
        print(month_list)
        folder_path = "/pbi/lb_bet/"
        local_folder_path = "./scripts/data/lb_bet/"
        for month in month_list:
            temp_df = bet_df.query(f"月日 == '{month}'")
            file_name = local_folder_path + month + ".csv"
            temp_df.to_csv(file_name, header=True, index=False)
            with open(file_name, 'rb') as f:
                self.dbx.files_upload(f.read(), folder_path + month + ".csv", mode=dropbox.files.WriteMode.overwrite)

    def export_raceuma_df(self):
        raceuma_df = self.raceuma_df[["競走コード", "馬番", "年月日", "得点", "馬券評価順位", "単勝配当", "複勝配当", "WIN_RATE", "JIKU_RATE", "ANA_RATE", "WIN_RANK", "JIKU_RANK", "ANA_RANK", "SCORE", "SCORE_RANK", "ck1", "ck2", "ck3"]].copy()
        raceuma_df.loc[:, "月日"] = raceuma_df["年月日"].apply(lambda x: str(x.year) + str(x.month))
        month_list = raceuma_df["月日"].drop_duplicates().tolist()
        print(month_list)
        folder_path = "/pbi/lb_raceuma/"
        local_folder_path = "./scripts/data/lb_raceuma/"
        for month in month_list:
            temp_df = raceuma_df.query(f"月日 == '{month}'")
            file_name = local_folder_path + month + ".csv"
            temp_df.to_csv(file_name, header=True, index=False)
            with open(file_name, 'rb') as f:
                self.dbx.files_upload(f.read(), folder_path + month + ".csv", mode=dropbox.files.WriteMode.overwrite)

    def export_race_df(self):
        race_df = self.race_df[["競走コード", "データ区分", "競走番号", "発走時刻", "月日", "場名", "距離"]].copy()
        race_df.loc[:, "月日"] = race_df["月日"].apply(lambda x: str(x.year) + str(x.month))
        month_list = race_df["月日"].drop_duplicates().tolist()
        print(month_list)
        folder_path = "/pbi/lb_race/"
        local_folder_path = "./scripts/data/lb_race/"
        for month in month_list:
            temp_df = race_df.query(f"月日 == '{month}'")
            file_name = local_folder_path + month + ".csv"
            temp_df.to_csv(file_name, header=True, index=False)
            with open(file_name, 'rb') as f:
                self.dbx.files_upload(f.read(), folder_path + month + ".csv", mode=dropbox.files.WriteMode.overwrite)

    def get_filterd_bet_df(self, n_value = 200):
        tansho_df = self.bet_df[self.bet_df["式別"] == 1].sort_values("日付").tail(n_value)
        umaren_df = self.bet_df[self.bet_df["式別"] == 5].sort_values("日付").tail(n_value)
        umatan_df = self.bet_df[self.bet_df["式別"] == 6].sort_values("日付").tail(n_value)
        wide_df = self.bet_df[self.bet_df["式別"] == 7].sort_values("日付").tail(n_value)
        sanrenpuku_df = self.bet_df[self.bet_df["式別"] == 8].sort_values("日付").tail(n_value)
        filter_bet_df = pd.concat([tansho_df, umaren_df, umatan_df, wide_df, sanrenpuku_df])
        return filter_bet_df

    def _get_bet_summary_df(self, bet_df):
        today_bet_df = bet_df.groupby("式別").sum()
        today_all_bet_df = today_bet_df.sum()
        today_all_bet_df.name = 0
        today_bet_df = today_bet_df.append(today_all_bet_df)
        today_bet_df.loc[:, "回収率"] = today_bet_df["結果"] / today_bet_df["金額"] * 100
        today_bet_df.reset_index(inplace=True)
        return today_bet_df

    def get_todays_bet_text(self):
        bet_text = '[ 本日結果 ] \r\n'
        today_bet_df = self._get_bet_summary_df(self.bet_df[self.bet_df["日付"] == self.end_date])
        bet_text += self._get_bet_text(today_bet_df)
        return bet_text

    def get_recent_bet_text(self):
        bet_text = '[ 直近結果 ] \r\n'
        recent_bet_df = self._get_bet_summary_df(self.bet_df)
        bet_text += self._get_bet_text(recent_bet_df)
        return bet_text

    def _get_avg_trend_text(self, type, td_avg, td_kpi, td_kpi_cnt, avg, kpi, kpi_cnt):
        td_avg = td_avg.astype(int)
        avg = avg.astype(int)
        rate_text = self._get_rate_text(td_kpi, kpi)
        trend_text = f' {type}平均：{td_avg:,}円({td_kpi}%, {td_kpi_cnt}件) {rate_text}\r\n {"―"*int(len(type))}平均：{avg:,}円({kpi}%)\r\n'
        return trend_text

    def _get_rate_text(self, kpi, base_kpi):
        if base_kpi != 0:
            rate = (kpi/base_kpi - 1) * 100 // 20
        else:
            rate = 0
        rate_text = "↑" if rate >= 0 else "↓"
        return rate_text * int(abs(rate))

    def _get_bet_text(self, bet_df):
        bet_text =''
        for index, row in bet_df.iterrows():
            baken_type = mu.trans_baken_type(row['式別'])
            return_rate = round(row['回収率']).astype(int)
            result_val = round(row['結果']).astype(int)
            bet_money = round(row['金額']).astype(int)
            bet_text += f'{baken_type} {return_rate}% ({result_val:,}円 / {bet_money:,}円)\r\n'
        return bet_text

    def _get_todays_df(self):
        race_df = self.race_df[self.race_df["月日"] == self.end_date][["データ区分", "場名", "発走時刻", "投票フラグ"]]
        end_date = self.end_date
        raceuma_df = self.raceuma_df.query("年月日 == @end_date & データ区分 == '7'")
        return race_df, raceuma_df

    def get_current_text(self):
        current_text = ''
        race_df, raceuma_df = self._get_todays_df()
        kaisai_text = str(set(race_df["場名"]))
        race_status = race_df["データ区分"].value_counts()
        all_race_count = ''
        for key, val in race_status.iteritems():
            all_race_count += 'St' + str(key) + ':' + str(val) + 'R | '
        self.final_race_time = race_df["発走時刻"].max()
        final_race = ' 最終レース:' + self.final_race_time.strftime('%H:%M')
        current_text += '[ 開催情報(' + dt.now().strftime('%Y/%m/%d %H') + '時点情報) ]\r\n'
        current_text += ' 開催場所: ' + kaisai_text + '\r\n'
        current_text += ' レース進捗：' + all_race_count + '\r\n'
        current_text += final_race + '\r\n'
        return current_text

    def get_trend_text(self):
        trend_text = ' レース配当トレンド ] \r\n'
        tansho_df = self.haraimodoshi_dict["tansho_df"]
        kpi_tansho_df = tansho_df.query("払戻 >= 1000")
        kpi_tansho = round(len(kpi_tansho_df)/ len(tansho_df) * 100 , 1)
        kpi_tansho_cnt = len(kpi_tansho_df)
        avg_tansho = round(tansho_df["払戻"].mean())

        td_tansho_df = self.todays_haraimodoshi_dict["tansho_df"]
        td_kpi_tansho_df = td_tansho_df.query("払戻 >= 1000")
        td_kpi_tansho = round(len(td_kpi_tansho_df)/ len(td_tansho_df) * 100 , 1)
        td_kpi_tansho_cnt = len(td_kpi_tansho_df)
        td_avg_tansho = round(td_tansho_df["払戻"].mean())
        trend_text += self._get_avg_trend_text("単勝", td_avg_tansho, td_kpi_tansho, td_kpi_tansho_cnt, avg_tansho, kpi_tansho, kpi_tansho_cnt)

        umaren_df = self.haraimodoshi_dict["umaren_df"]
        kpi_umaren_df = umaren_df.query("払戻 >= 5000")
        kpi_umaren = round(len(kpi_umaren_df)/ len(umaren_df) * 100 , 1)
        kpi_umaren_cnt = len(kpi_umaren_df)
        avg_umaren = round(umaren_df["払戻"].mean())
        td_umaren_df = self.todays_haraimodoshi_dict["umaren_df"]
        td_kpi_umaren_df = td_umaren_df.query("払戻 >= 5000")
        td_kpi_umaren = round(len(td_kpi_umaren_df)/ len(td_umaren_df) * 100 , 1)
        td_kpi_umaren_cnt = len(td_kpi_umaren_df)
        td_avg_umaren = round(td_umaren_df["払戻"].mean())
        trend_text += self._get_avg_trend_text("馬連", td_avg_umaren, td_kpi_umaren, td_kpi_umaren_cnt, avg_umaren, kpi_umaren, kpi_umaren_cnt)

        umatan_df = self.haraimodoshi_dict["umatan_df"]
        kpi_umatan_df = umatan_df.query("払戻 >= 5000")
        kpi_umatan = round(len(kpi_umatan_df) / len(umatan_df) * 100, 1)
        kpi_umatan_cnt = len(kpi_umatan_df)
        avg_umatan = round(umatan_df["払戻"].mean())
        td_umatan_df = self.todays_haraimodoshi_dict["umatan_df"]
        td_kpi_umatan_df = td_umatan_df.query("払戻 >= 5000")
        td_kpi_umatan = round(len(td_kpi_umatan_df) / len(td_umatan_df) * 100, 1)
        td_kpi_umatan_cnt = len(td_kpi_umatan_df)
        td_avg_umatan = round(td_umatan_df["払戻"].mean())
        trend_text += self._get_avg_trend_text("馬単", td_avg_umatan, td_kpi_umatan, td_kpi_umatan_cnt, avg_umatan, kpi_umatan, kpi_umatan_cnt)

        wide_df = self.haraimodoshi_dict["wide_df"]
        kpi_wide_df = wide_df.query("払戻 >= 3000")
        kpi_wide = round(len(kpi_wide_df) / len(wide_df) * 100, 1)
        kpi_wide_cnt = len(kpi_wide_df)
        avg_wide = round(wide_df["払戻"].mean())
        td_wide_df = self.todays_haraimodoshi_dict["wide_df"]
        td_kpi_wide_df = td_wide_df.query("払戻 >= 3500")
        td_kpi_wide = round(len(td_kpi_wide_df) / len(td_wide_df) * 100, 1)
        td_kpi_wide_cnt = len(td_kpi_wide_df)
        td_avg_wide = round(td_wide_df["払戻"].mean())
        trend_text += self._get_avg_trend_text("ワイド", td_avg_wide, td_kpi_wide, td_kpi_wide_cnt, avg_wide, kpi_wide, kpi_wide_cnt)

        sanrenpuku_df = self.haraimodoshi_dict["sanrenpuku_df"]
        kpi_sanrenpuku_df = sanrenpuku_df.query("払戻 >= 7500")
        kpi_sanrenpuku = round(len(kpi_sanrenpuku_df) / len(sanrenpuku_df) * 100, 1)
        kpi_sanrenpuku_cnt = len(kpi_sanrenpuku_df)
        avg_sanrenpuku = round(sanrenpuku_df["払戻"].mean())
        td_sanrenpuku_df = self.todays_haraimodoshi_dict["sanrenpuku_df"]
        td_kpi_sanrenpuku_df = td_sanrenpuku_df.query("払戻 >= 7500")
        td_kpi_sanrenpuku = round(len(td_kpi_sanrenpuku_df) / len(td_sanrenpuku_df) * 100, 1)
        td_kpi_sanrenpuku_cnt = len(td_kpi_sanrenpuku_df)
        td_avg_sanrenpuku = round(td_sanrenpuku_df["払戻"].mean())
        trend_text += self._get_avg_trend_text("三連複", td_avg_sanrenpuku, td_kpi_sanrenpuku, td_kpi_sanrenpuku_cnt, avg_sanrenpuku, kpi_sanrenpuku, kpi_sanrenpuku_cnt)

        return trend_text

    def get_kaime_target_text(self):
        target_text = '[ 軸候補結果 ]\r\n'
        race_df, raceuma_df = self._get_todays_df()
        query_umaren1 = "得点 >= 48 and SCORE_RANK <= 5 and 馬券評価順位 <= 2 and デフォルト得点順位 <= 4 and 予想人気 <= 8 and SCORE >= 49 and JIKU_RATE >= 44 and WIN_RATE >= 50"
        query_umatan_1 = "馬券評価順位 <= 2 and SCORE >= 51 and デフォルト得点 >= 47 and JIKU_RATE >= 43 and WIN_RATE >= 49 and ANA_RATE >= 43 and ANA_RATE < 60"
        query_umatan_2 = "馬券評価順位 <= 3 and SCORE_RANK <= 6 and デフォルト得点順位 <= 4 and 得点 >= 47 and SCORE >= 43 and デフォルト得点 >= 45 and 得点V3 >= 43 and 得点V3 < 55 and WIN_RATE >= 40 and ANA_RATE >= 40 and ANA_RATE < 60"
        query_wide_1 = "馬券評価順位 <= 3 and 予想人気 <= 4 and 得点 >= 47 and SCORE_RANK >= 2 and SCORE_RANK < 9 and SCORE >= 47 and デフォルト得点 >= 41 and WIN_RATE >= 41 and WIN_RATE < 60 and JIKU_RATE >= 40"
        query_sanrenpuku_1 ="馬券評価順位 <= 4 and 得点 >= 50 and デフォルト得点 >= 50 and SCORE >= 44 and 予想人気 >= 2 and 予想人気 < 9 and WIN_RATE >= 47 and JIKU_RATE >= 42 and ANA_RATE <= 58"
        umaren1_df = raceuma_df.query(query_umaren1)
        target_text += "馬連軸：" + self._calc_raceuma_target_result(umaren1_df, "ren")
        umatan1_df = raceuma_df.query(query_umatan_1)
        target_text += "馬単軸1：" + self._calc_raceuma_target_result(umatan1_df, "ck1")
        umatan2_df = raceuma_df.query(query_umatan_2)
        target_text += "馬単軸2：" + self._calc_raceuma_target_result(umatan2_df, "ck2")
        wide1_df = raceuma_df.query(query_wide_1)
        target_text += "ワイ軸：" + self._calc_raceuma_target_result(wide1_df, "fuku")
        sanrenpuku1_df = raceuma_df.query(query_sanrenpuku_1)
        target_text += "三複軸：" + self._calc_raceuma_target_result(sanrenpuku1_df, "fuku")
        return target_text

    def get_summary_text(self):
        summary_text = '[ KPI集計結果 ] \r\n'
        race_df, raceuma_df = self._get_todays_df()
        score_raceuma_df = raceuma_df.query("馬券評価順位 == 1")
        default_raceuma_df = raceuma_df.query("デフォルト得点順位 == 1")
        ninki_raceuma_df = raceuma_df.query("単勝人気 == 1")

        total_score_raceuma_df = self.raceuma_df.query("馬券評価順位 == 1")
        total_default_raceuma_df = self.raceuma_df.query("デフォルト得点順位 == 1")
        total_ninki_raceuma_df = self.raceuma_df.query("単勝人気 == 1")
        score_result_txt = self._calc_raceuma_result(score_raceuma_df, total_score_raceuma_df)
        default_result_txt = self._calc_raceuma_result(default_raceuma_df, total_default_raceuma_df)
        ninki_result_txt = self._calc_raceuma_result(ninki_raceuma_df, total_ninki_raceuma_df)

        summary_text += '馬券評価順位１位' + score_result_txt
        summary_text += 'デフォルト得点１位' + default_result_txt
        summary_text += '一番人気' + ninki_result_txt + "\r\n"
        return summary_text

    def _calc_raceuma_target_result(self, df, type):
        summary_df = df.describe()
        sum_df = df[["ck1", "ck2", "ck3", "ckg"]].sum()
        tansho_return = round(summary_df["単勝配当"]["mean"], 1)
        fukusho_return = round(summary_df["複勝配当"]["mean"], 1)
        chaku_text = str(sum_df["ck1"])
        total_count = sum_df["ck1"]
        for key, val in sum_df.iteritems():
            if key != 'ck1':
                chaku_text += '-' + str(val)
                total_count += val
        if total_count != 0:
            ck1_rate = (sum_df["ck1"] / total_count) * 100
            ck2_rate = (sum_df["ck2"] / total_count) * 100
            ren_rate = ((sum_df["ck1"]  + sum_df["ck2"]) / total_count) * 100
            fuku_rate = ((sum_df["ck1"] + sum_df["ck2"] + sum_df["ck3"]) / total_count) * 100
        else:
            ck1_rate = 0; ck2_rate =0; ren_rate =0; fuku_rate =0;
        if type == "ren":
            target_text = "連：" + str(int(ren_rate)) + "%"
        elif type == "ck1":
            target_text = "１着：" + str(int(ck1_rate)) + "%"
        elif type == "ck2":
            target_text = "２着：" + str(int(ck2_rate)) + "%"
        elif type == "fuku":
            target_text = "複：" + str(int(fuku_rate)) + "%"
        else:
            target_text = ""

        res_text = f' ({chaku_text}) {target_text} \r\n  単：{tansho_return}% 複：{fukusho_return}%\r\n'
        return res_text

    def _calc_raceuma_result(self, df, total_df):
        summary_df = df.describe()
        sum_df = df[["ck1", "ck2", "ck3", "ckg"]].sum()
        av_ninki = round(summary_df["単勝人気"]["mean"], 1)
        av_chakujn = round(summary_df["確定着順"]["mean"], 1)
        tansho_return = round(summary_df["単勝配当"]["mean"], 1)
        fukusho_return = round(summary_df["複勝配当"]["mean"], 1)
        chaku_text = str(sum_df["ck1"])
        for key, val in sum_df.iteritems():
            if key != 'ck1':
                chaku_text += '-' + str(val)

        t_summary_df = total_df.describe()
        t_av_ninki = round(t_summary_df["単勝人気"]["mean"], 1)
        t_av_chakujn = round(t_summary_df["確定着順"]["mean"], 1)
        t_tansho_return = round(t_summary_df["単勝配当"]["mean"], 1)
        t_fukusho_return = round(t_summary_df["複勝配当"]["mean"], 1)

        tansho_rate_text = self._get_rate_text(tansho_return, t_tansho_return)
        fukusho_rate_text = self._get_rate_text(fukusho_return, t_fukusho_return)

        res_text = f' ({chaku_text})\r\n Av：{av_chakujn}着 単：{tansho_return}%({tansho_rate_text}) 複：{fukusho_return}%({fukusho_rate_text})\r\n'
        res_text += f' Av：{t_av_chakujn}着 単：{t_tansho_return}% 複：{t_fukusho_return}%\r\n'
        return res_text
