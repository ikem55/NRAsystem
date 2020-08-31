import azure.cosmos.cosmos_client as cosmos_client
import my_config as mc
import pandas as pd

#pd.set_option('display.max_columns', 3000)
#pd.set_option('display.max_rows', 3000)

from modules.extract import Extract
import modules.util as mu

class Import_to_CosmosDB(object):
    def __init__(self, target_date, test_flag):
        self.target_date = target_date
        self.config = mc.return_cosmos_info(test_flag)
        self.client = cosmos_client.CosmosClient(url = self.config["ENDPOINT"],
                                                 credential={'masterKey': self.config["PRIMARYKEY"]})
        self.database = self.client.get_database_client(self.config["DATABASE"])
        container_prefix = self.config["CONTAINER"]
        self.container = self.database.get_container_client(container_prefix + "baoz")
        self.ext = Extract(target_date, target_date, False)
        self.race_dict = {"競走コード": "RK", "月日": "target_date", "距離": "K", "競走番号": "RN", "場名": "BN",
                          "発走時刻": "HJ", "データ区分": "DC"}
        self.raceuma_dict = {"データ区分": "DC", "競走コード": "RK", "馬番": "UM", "年月日": "target_date", "予想タイム指数順位": "RT",
                             "単勝配当": "TAN", "複勝配当": "FKU", "単勝人気": "TN", "単勝オッズ": "TO", "予想人気": "YN",
                             "異常区分コード": "IC", "確定着順": "CK", "デフォルト得点順位": "RD", "WIN_RATE": "WR",
                             "JIKU_RATE": "JR", "ANA_RATE": "AR", "WIN_RANK": "WO", "JIKU_RANK": "WO", "ANA_RANK": "AO",
                             "SCORE": "SC", "SCORE_RANK": "SR", "WIN_SCORE": "WS", "JIKU_SCORE": "JS", "ANA_SCORE": "AS"}
        self.bet_df = {"競走コード": "RK", "式別": "S", "結果": "K", "金額": "M", "馬番": "UB"}
        self.haraimodoshi_df = {"競走コード": "RK", "馬番": "UB", "払戻": "H"}

    def upsert_df(self, df):
        # https://docs.microsoft.com/ja-jp/python/api/azure-cosmos/azure.cosmos.containerproxy?view=azure-python#read-item-item--partition-key--populate-query-metrics-none--post-trigger-include-none----kwargs-
        for index, row in df.iterrows():
            dict = row.to_dict()
            #print(dict)
            self.container.upsert_item(dict)

    def get_data(self, type, target_date):
        query = f"SELECT * from c WHERE c.type = '{type}' and c.target_date = '{target_date}'"
        items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        dflist = []
        for item in items:
            dflist.append(dict(item))
        df = pd.DataFrame(dflist)
        column_dict = self._decode_columns(type)
        df = df.rename(columns=column_dict)
        return df

    def _decode_columns(self, type):
        if type == "race": columns = self.raceuma_dict
        elif type == "raceuma": columns =self.raceuma_dict
        elif type in ["単勝", "複勝", "馬連", "馬単", "ワイド", "三連複"]: columns =self.haraimodoshi_df
        elif type in ["馬券 ", "仮想"]: columns = self.bet_df
        else: columns = {}
        d_swap = {v: k for k, v in columns.items()}
        return d_swap

    def import_predict_data(self):
        race_df = self.ext.get_race_table_base().query("データ区分 == '7'").copy()
        race_df = race_df[["競走コード", "月日", "距離", "競走番号", "場名", "発走時刻", "データ区分"]]
        race_df.loc[:, "月日"] = race_df["月日"].apply(lambda x: x.strftime('%Y-%m-%d'))
        race_df.loc[:, "発走時刻"] = race_df["発走時刻"].apply(lambda x: x.strftime('%H:%M'))
        race_df.loc[:, "type"] = "race"
        race_df.loc[:, "id"] = race_df["競走コード"].astype("str")
        race_df.rename(columns=self.race_dict, inplace=True)
        print(race_df.shape)
        self.upsert_df(race_df)

        raceuma_df = self.ext.get_raceuma_table_base().query("データ区分 == '7'").copy().fillna(0)
        raceuma_df = raceuma_df[["データ区分", "競走コード", "馬番", "年月日", "予想タイム指数順位", "単勝配当", "複勝配当", "単勝人気", "単勝オッズ",
                                 "予想人気" , "異常区分コード", "確定着順", "デフォルト得点順位", "WIN_RATE", "JIKU_RATE",
                                 "ANA_RATE", "WIN_RANK", "JIKU_RANK", "ANA_RANK", "SCORE", "SCORE_RANK", "WIN_SCORE",
                                 "JIKU_SCORE", "ANA_SCORE"]]
        raceuma_df.loc[:, "年月日"] = raceuma_df["年月日"].apply(lambda x: x.strftime('%Y-%m-%d'))
        raceuma_df.loc[:, "type"] = "raceuma"
        raceuma_df.loc[:, "id"] = raceuma_df["競走コード"].astype("str") + raceuma_df["馬番"].astype("str")
        raceuma_df.rename(columns=self.raceuma_dict, inplace=True)
        print(raceuma_df.shape)
        self.upsert_df(raceuma_df)

        base_haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        haraimodoshi_dict = mu.get_haraimodoshi_dict(base_haraimodoshi_df)

        tansho_df = haraimodoshi_dict["tansho_df"]
        tansho_df.loc[:, "target_date"] = self.target_date
        tansho_df.loc[:, "type"] = "単勝"
        tansho_df.loc[:, "id"] = "T" + tansho_df["競走コード"].astype("str") + tansho_df["馬番"].astype("str")
        tansho_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(tansho_df.shape)
        self.upsert_df(tansho_df)

        fukusho_df = haraimodoshi_dict["fukusho_df"]
        fukusho_df.loc[:, "target_date"] = self.target_date
        fukusho_df.loc[:, "type"] = "複勝"
        fukusho_df.loc[:, "id"] = "F" + fukusho_df["競走コード"].astype("str") + fukusho_df["馬番"].astype("str")
        fukusho_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(fukusho_df.shape)
        self.upsert_df(fukusho_df)

        umaren_df = haraimodoshi_dict["umaren_df"]
        umaren_df.loc[:, "target_date"] = self.target_date
        umaren_df.loc[:, "type"] = "馬連"
        umaren_df.loc[:, "index"] = umaren_df["馬番"].apply(lambda x: "_".join(map(str, x)))
        umaren_df.loc[:, "id"] = "UR" + umaren_df["競走コード"].astype("str") + umaren_df["index"].astype("str")
        umaren_df.drop("index", axis=1, inplace=True)
        umaren_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(umaren_df.shape)
        self.upsert_df(umaren_df)

        umatan_df = haraimodoshi_dict["umatan_df"]
        umatan_df.loc[:, "target_date"] = self.target_date
        umatan_df.loc[:, "type"] = "馬単"
        umatan_df.loc[:, "index"] = umatan_df["馬番"].apply(lambda x: "_".join(map(str, x)))
        umatan_df.loc[:, "id"] = "UT" + umatan_df["競走コード"].astype("str") + umatan_df["index"].astype("str")
        umatan_df.drop("index", axis=1, inplace=True)
        umatan_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(umatan_df.shape)
        self.upsert_df(umatan_df)

        wide_df = haraimodoshi_dict["wide_df"]
        wide_df.loc[:, "target_date"] = self.target_date
        wide_df.loc[:, "type"] = "ワイド"
        wide_df.loc[:, "index"] = wide_df["馬番"].apply(lambda x: "_".join(map(str, x)))
        wide_df.loc[:, "id"] = "W" + wide_df["競走コード"].astype("str") + wide_df["index"].astype("str")
        wide_df.drop("index", axis=1, inplace=True)
        wide_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(wide_df.shape)
        self.upsert_df(wide_df)

        sanrenpuku_df = haraimodoshi_dict["sanrenpuku_df"]
        sanrenpuku_df.loc[:, "target_date"] = self.target_date
        sanrenpuku_df.loc[:, "type"] = "３連複"
        sanrenpuku_df.loc[:, "index"] = sanrenpuku_df["馬番"].apply(lambda x: "_".join(map(str, x)))
        sanrenpuku_df.loc[:, "id"] = "S" + sanrenpuku_df["競走コード"].astype("str") + sanrenpuku_df["index"].astype("str")
        sanrenpuku_df.drop("index", axis=1, inplace=True)
        sanrenpuku_df.rename(columns=self.haraimodoshi_df, inplace=True)
        print(sanrenpuku_df.shape)
        self.upsert_df(sanrenpuku_df)

        base_bet_df = self.ext.get_bet_table_base()
        bet_df = base_bet_df.copy()
        bet_df.loc[:, "index"] = bet_df["番号"].apply(lambda x: str(x).zfill(6))
        bet_df.loc[:, "馬番"] = bet_df["番号"].apply(lambda x: x if x <= 20 else mu.separate_umaban(x))
        bet_df.loc[:, "結果"] = bet_df["結果"] * bet_df["金額"] / 100
        bet_df.loc[:, "式別"] = bet_df["式別"].apply(lambda x: mu.trans_baken_type(x))
        bet_df.loc[:, "id"] = bet_df["競走コード"].astype("str") + bet_df["index"].astype("str")
        bet_df.loc[:, "target_date"] = bet_df["日付"].apply(lambda x: x.strftime('%Y-%m-%d'))
        bet_df.loc[:, "type"] = "馬券"
        bet_df = bet_df[["id", "競走コード", "式別", "target_date", "結果", "金額", "type", "馬番"]]
        bet_df.rename(columns=self.bet_df, inplace=True)
        print(bet_df.shape)
        self.upsert_df(bet_df)

        base_vbet_df = self.ext.get_vbet_table_base()
        vbet_df = base_vbet_df.copy()
        vbet_df.loc[:, "index"] = vbet_df["番号"].apply(lambda x: str(x).zfill(6))
        vbet_df.loc[:, "馬番"] = vbet_df["番号"].apply(lambda x: x if x <= 20 else mu.separate_umaban(x))
        vbet_df.loc[:, "結果"] = vbet_df["結果"] * vbet_df["金額"] / 100
        vbet_df.loc[:, "式別"] = vbet_df["式別"].apply(lambda x: mu.trans_baken_type(x))
        vbet_df.loc[:, "id"] = vbet_df["競走コード"].astype("str") + vbet_df["index"].astype("str")
        vbet_df.loc[:, "target_date"] = vbet_df["日付"].apply(lambda x: x.strftime('%Y-%m-%d'))
        vbet_df.loc[:, "type"] = "仮想"
        vbet_df = vbet_df[["id", "競走コード", "式別", "target_date", "結果", "金額", "type", "馬番"]]
        vbet_df.rename(columns=self.bet_df, inplace=True)
        print(vbet_df.shape)
        self.upsert_df(vbet_df)
