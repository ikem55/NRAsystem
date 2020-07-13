import pandas as pd
import pyodbc

class Extract(object):
    """
    データ抽出に関する共通モデル。データ取得については下位モデルで定義する。
    """
    mock_flag = False
    """ mockデータを使うかの判断に使用するフラグ。Trueの場合はMockデータを使う """
    mock_path = '../mock_data/lb/'
    """ mockファイルが格納されているフォルダのパス """
    row_cnt = 1000

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        if mock_flag:
            self.set_mock_path()
            self.mock_flag = mock_flag

    def set_mock_path(self):
        """ mock_flagをTrueにしてmockのパスを設定する。  """
        self.mock_path_race = self.mock_path + 'race.pkl'
        self.mock_path_raceuma = self.mock_path + 'raceuma.pkl'
        self.mock_path_bet = self.mock_path + 'bet.pkl'
        self.mock_path_haraimodoshi = self.mock_path + 'haraimodoshi.pkl'
        self.mock_path_zandaka = self.mock_path + 'zandaka.pkl'
        self.mock_path_horse = self.mock_path + 'horse.pkl'
        self.mock_path_mydb = self.mock_path + 'mydb.pkl'
        self.mock_path_tansho = self.mock_path + 'tansho.pkl'
        self.mock_path_fukusho = self.mock_path + 'fukusho.pkl'
        self.mock_path_umaren = self.mock_path + 'umaren.pkl'
        self.mock_path_umatan = self.mock_path + 'umatan.pkl'
        self.mock_path_wide = self.mock_path + 'wide.pkl'
        self.mock_path_sanrenpuku = self.mock_path + 'sanrenpuku.pkl'

    def create_mock_data(self):
        """ mock dataを作成する  """
        self.mock_flag = False
        race_df = self.get_race_table_base()
        raceuma_df = self.get_raceuma_table_base()
        bet_df = self.get_bet_table_base()
        haraimodoshi_df = self.get_haraimodoshi_table_base()
        zandaka_df = self.get_zandaka_table_base()
        horse_df = self.get_horse_table_base()
        mydb_df = self.get_mydb_table_base()
        tansho_df = self.get_tansho_table_base()
        fukusho_df = self.get_fukusho_table_base()
        umaren_df = self.get_umaren_table_base()
        umatan_df = self.get_umatan_table_base()
        wide_df = self.get_wide_table_base()
        sanrenpuku_df = self.get_sanrenpuku_table_base()

        self.set_mock_path()
        race_df.to_pickle(self.mock_path_race)
        raceuma_df.to_pickle(self.mock_path_raceuma)
        bet_df.to_pickle(self.mock_path_bet)
        haraimodoshi_df.to_pickle(self.mock_path_haraimodoshi)
        zandaka_df.to_pickle(self.mock_path_zandaka)
        horse_df.to_pickle(self.mock_path_horse)
        mydb_df.to_pickle(self.mock_path_mydb)
        tansho_df.to_pickle(self.mock_path_tansho)
        fukusho_df.to_pickle(self.mock_path_fukusho)
        umaren_df.to_pickle(self.mock_path_umaren)
        umatan_df.to_pickle(self.mock_path_umatan)
        wide_df.to_pickle(self.mock_path_wide)
        sanrenpuku_df.to_pickle(self.mock_path_sanrenpuku)


    def get_race_table_for_view(self):
        """ 馬王ZのMDGからレーステーブルからデータを取得する。。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_race)
        else:
            cnxn = self._connect_baoz_mdb()
            select_sql = 'SELECT * FROM レースT WHERE 月日 >= #' + \
                self.start_date + '# AND 月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype({'トラック種別コード': object, '主催者コード': object, '場コード': object, '競走種別コード': object, '競走条件コード': object, 'トラックコード': object,
                                '天候コード': object, '馬場状態コード': object, '投票フラグ': object, '波乱度': object, '馬券発売フラグ': object, '予想計算状況フラグ': object})
        return_df = df[df["主催者コード"] == 2].copy()
        return return_df

    def get_race_table_base(self):
        """ 馬王ZのMDGからレーステーブルからデータを取得する。。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_race)
        else:
            cnxn = self._connect_baoz_mdb()
#            select_sql = 'SELECT データ区分, 競走コード, 月日, 距離, トラック種別コード, 主催者コード, 競走番号, 場コード, 場名, グレードコード, 競走種別コード, 競走条件コード, 発走時刻, 頭数, 天候コード, 前３ハロン, 前４ハロン' \
#                         ', 後３ハロン, 後４ハロン, トラックコード,  馬場状態コード, 前半タイム, 予想計算済み, 予想勝ち指数, ペース, 初出走頭数, 混合, 予想決着指数, 投票フラグ, 波乱度, 馬券発売フラグ, 予想計算状況フラグ, メインレース, タイム指数誤差, 登録頭数, 回次, 日次 FROM レースT WHERE 月日 >= #' + \
#                self.start_date + '# AND 月日 <= #' + self.end_date + '#'
            select_sql = 'SELECT * FROM レースT WHERE 月日 >= #' + \
                self.start_date + '# AND 月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype({'トラック種別コード': object, '主催者コード': object, '場コード': object, '競走種別コード': object, '競走条件コード': object, 'トラックコード': object,
                                '天候コード': object, '馬場状態コード': object, '投票フラグ': object, '波乱度': object, '馬券発売フラグ': object, '予想計算状況フラグ': object})
        return_df = df[df["主催者コード"] == 2].copy()
        return return_df

    def get_race_before_table_base(self):
        temp_df = self.get_race_table_base()
        df = temp_df.astype({'場コード': object, '競走種別コード': object, 'トラックコード': object})
        return df[["データ区分", "競走コード", "月日", "距離", "トラック種別コード", "主催者コード", "競走番号", "場コード", "場名", "グレードコード", "競走種別コード", "競走条件コード", "発走時刻", "頭数", "トラックコード", "予想勝ち指数", "初出走頭数", "混合", "予想決着指数", "登録頭数", "回次", "日次"]].copy()

    def get_raceuma_table_base(self):
        """  出走馬テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_raceuma)
        else:
            cnxn = self._connect_baoz_ex_mdb()
            select_sql = 'SELECT * FROM 出走馬T WHERE 年月日 >= #' + \
                self.start_date + '# AND 年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            race_df = self.get_race_table_base()["競走コード"]
            temp_df = pd.merge(df_org, race_df, on="競走コード")
            df = temp_df.astype({'血統登録番号': object, '性別コード': object, '展開コード': object, '騎手コード': object, '騎手所属場コード': object,
                                '見習区分': object, '調教師コード': object, '調教師所属場コード': object, '異常区分コード': object, '前走トラック種別コード':object})
        return df

    def get_raceuma_before_table_base(self):
        temp_df = self.get_raceuma_table_base()
        return temp_df[["データ区分", "データ作成年月日", "競走コード", "馬番", "枠番", "血統登録番号", "性別コード", "年月日", "予想タイム指数", "予想タイム指数順位", "デフォルト得点", "近走競走コード1", "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4", "近走競走コード5", "近走馬番5", "休養週数"
            , "休養後出走回数", "予想オッズ", "予想人気", "血統距離評価", "血統トラック評価", "血統成長力評価", "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B", "先行指数", "先行指数順位", "予想展開", "クラス変動", "騎手コード", "騎手所属場コード", "見習区分", "騎手名", "テン乗り", "騎手評価", "調教師評価", "枠順評価", "脚質評価"
            , "キャリア", "馬齢", "調教師コード", "調教師所属場コード", "調教師名", "負担重量", "距離増減", "前走着順", "前走人気", "前走着差", "前走トラック種別コード", "前走馬体重", "前走頭数", "タイム指数上昇係数", "タイム指数回帰推定値", "タイム指数回帰標準偏差", "所属",  "転厩", "斤量比", "前走休養週数", "騎手ランキング", "調教師ランキング", "得点V1", "得点V2"
            , "得点V3", "得点V1順位", "得点V2順位", "デフォルト得点順位", "得点V3順位"]].copy()

    def get_horse_table_base(self):
        """ 競走馬マスタからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_horse)
        else:
            cnxn = self._connect_baoz_ra_mdb()
            select_sql = 'SELECT * FROM 競走馬マスタ WHERE データ作成年月日 >= #2016/01/01#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype({'血統登録番号': object, '競走馬抹消区分': object, 'JRA施設在厩フラグ': object, '馬記号コード': object, '性別コード': object, '毛色コード': object, '繁殖登録番号1': object, '繁殖登録番号2': object, '繁殖登録番号3': object, '繁殖登録番号4': object, '繁殖登録番号5': object, '繁殖登録番号6': object,
                                '繁殖登録番号7': object, '繁殖登録番号8': object, '繁殖登録番号9': object, '繁殖登録番号10': object, '繁殖登録番号11': object, '繁殖登録番号12': object, '繁殖登録番号13': object, '繁殖登録番号14': object, '東西所属コード': object, '調教師コード': object, '生産者コード': object, '馬主コード': object})
        return df

    def get_tansho_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_tansho)
        else:
            cnxn = self._connect_baoz_o1_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, 単勝全オッズ, 単勝票数合計 FROM 単複枠オッズT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'単勝全オッズ':'全オッズ', '単勝票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})# (オッズ4桁999.9倍で設定)*繰り返し28、データ区分　2: 前日売最終 4:確定 5:確定(月曜) 9:レース中止 10:該当レコード削除(提供ミスなどの理由による)
        return df

    def get_fukusho_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_fukusho)
        else:
            cnxn = self._connect_baoz_o1_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, 複勝全オッズ, 複勝票数合計 FROM 単複枠オッズT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'複勝全オッズ':'全オッズ', '複勝票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})# (最低オッズ4桁999.9倍で設定・最高オッズ4桁)*繰り返し28
        return df

    def get_umaren_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_umaren)
        else:
            cnxn = self._connect_baoz_o2_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, 馬連全オッズ, 馬連票数合計 FROM 馬連オッズT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'馬連全オッズ':'全オッズ', '馬連票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})# (オッズ6桁99999.9倍で設定)*繰り返し153(18!)
        return df

    def get_umatan_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_umatan)
        else:
            cnxn = self._connect_baoz_o4_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, 馬単全オッズ, 馬単票数合計 FROM 馬単オッズT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'馬単全オッズ':'全オッズ', '馬単票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})
        return df

    def get_wide_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_wide)
        else:
            cnxn = self._connect_baoz_o3_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, ワイド全オッズ, ワイド票数合計 FROM ワイドオッズT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'ワイド全オッズ':'全オッズ', 'ワイド票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})#(連番4桁・最低オッズ5桁9999.9倍で設定・最高オッズ5桁・人気3桁)*繰り返し153
        return df

    def get_sanrenpuku_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_sanrenpuku)
        else:
            cnxn = self._connect_baoz_o5n_mdb()
            select_sql = 'SELECT データ区分, データ作成年月日, 競走コード, 登録頭数, 三連複全オッズ, 三連複票数合計 FROM 三連複オッズNT WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.rename(columns={'三連複全オッズ':'全オッズ', '三連複票数合計': '票数合計'}).astype({'データ区分': object, '全オッズ': object})#(オッズ6桁99999.9倍で設定)*繰り返し816
        return df

    def get_haraimodoshi_table_base(self):
        """ 払戻テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return:dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_haraimodoshi)
        else:
            cnxn = self._connect_baoz_ra_mdb()
            select_sql = 'SELECT * FROM 払戻T WHERE データ作成年月日 >= #' + \
                self.start_date + '# AND データ作成年月日 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype({'不成立フラグ': object, '特払フラグ': object, '返還フラグ': object, '返還馬番情報': object, '返還枠番情報': object, '返還同枠情報': object, '単勝馬番1': object, '単勝馬番2': object, '単勝馬番3': object, '複勝馬番1': object, '複勝馬番2': object, '複勝馬番3': object, '複勝馬番4': object, '複勝馬番5': object, '枠連連番1': object, '枠連連番2': object, '枠連連番3': object, '馬連連番1': object, '馬連連番2': object, '馬連連番3': object, 'ワイド連番1': object, 'ワイド連番2': object,
                                'ワイド連番3': object, 'ワイド連番4': object, 'ワイド連番5': object, 'ワイド連番6': object, 'ワイド連番7': object, '枠単連番1': object, '枠単連番2': object, '枠単連番3': object, '馬単連番1': object, '馬単連番2': object, '馬単連番3': object, '馬単連番4': object, '馬単連番5': object, '馬単連番6': object, '三連複連番1': object, '三連複連番2': object, '三連複連番3': object, '三連単連番1': object, '三連単連番2': object, '三連単連番3': object, '三連単連番4': object, '三連単連番5': object, '三連単連番6': object})
        return df

    def get_bet_table_base(self):
        """ 投票記録テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_bet)
        else:
            cnxn = self._connect_baoz_bet_mdb()
            select_sql = 'SELECT * FROM 投票記録T WHERE 日付 >= #' + \
                self.start_date + '# AND 日付 <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype(
                {'式別': object, 'レース種別': object, 'PAT_ID': object, '投票方法': object})
        return df

    def get_zandaka_table_base(self):
        """ 残高テーブルからデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_zandaka)
        else:
            cnxn = self._connect_baoz_mdb()
            select_sql = 'SELECT * FROM 残高T'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype({'主催者コード': object})
        return df

    def get_mydb_table_base(self):
        """ 地方競馬テーブル（自分で作成したデータ）からデータを取得する。mock_flagがTrueの時はmockデータを取得する。

        :return: dataframe
        """
        if self.mock_flag:
            df = pd.read_pickle(self.mock_path_mydb)
        else:
            cnxn = self._connect_baoz_my_mdb()
            select_sql = 'SELECT * FROM 地方競馬レース馬V1 WHERE target_date >= #' + \
                self.start_date + '# AND target_date <= #' + self.end_date + '#'
            df_org = pd.read_sql(select_sql, cnxn)
            cnxn.close()
            df = df_org.astype(
                {'target': object, 'target_date': object})
        return df

    def _connect_baoz_mdb(self):
        """ BaoZ.mdbとの接続をする。レースT,残高Tとの接続に使用

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\Baoz.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_ex_mdb(self):
        """ BaoZ-ex.mdbとの接続をする。競走馬Tとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\Baoz.ex.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_bet_mdb(self):
        """ BaoZ-Bet.mdbとの接続をする。投票記録Tとの接続に使用

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-Bet.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_ra_mdb(self):
        """ BaoZ-RA.mdbとの接続をする。競走馬マスタ、払戻Tとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-RA.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_o1_mdb(self):
        """ BaoZ-O1.mdbとの接続をする。単複枠オッズT・枠単オッズTとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O1.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_o2_mdb(self):
        """ BaoZ-O2.mdbとの接続をする。馬連オッズTとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O2.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_o3_mdb(self):
        """ BaoZ-O3.mdbとの接続をする。ワイドオッズTとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O3.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_o4_mdb(self):
        """ BaoZ-O4.mdbとの接続をする。馬単オッズTとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O4.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_o5n_mdb(self):
        """ BaoZ-O5N.mdbとの接続をする。三連複オッズNTとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O5N.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn

    def _connect_baoz_my_mdb(self):
        """ MyDB.mdbとの接続をする。地方競馬Tとの接続に使用。

        :return: cnxn
        """
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
        )
        cnxn = pyodbc.connect(conn_str)
        return cnxn
