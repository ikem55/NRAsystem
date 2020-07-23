from modules.load import Load
from modules.sk_model import SkModel
import modules.util as mu
import my_config as mc

import pandas as pd
import os
import numpy as np
import pickle
import category_encoders as ce
import pyodbc

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE

import optuna.integration.lightgbm as lgb
import lightgbm as lgb_original

class SkProc(object):
    """
    機械学習を行うのに必要なプロセスを定義する。learning/predictデータの作成プロセスや
    """
    learning_df = ""
    categ_columns = ""
    target_enc_columns = ""
    target_flag = ""
    x_df = ""
    y_df = ""
    X_train = ""
    X_test = ""
    y_train = ""
    y_test = ""
    label_list = ""
    table_name = 'race_win'
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
    )

    def __init__(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.version_str = version_str # type:haito, win etc.
        self.model_name = model_name #race or raceuma
        self._set_folder_path(version_str, model_name, test_flag)
        self._set_index_list(model_name)
        self._set_obj_column_list(version_str)
        self._set_test_table(test_flag)
        mu.create_folder(self.model_folder)
        self.ld = self._get_load_object(version_str, start_date, end_date, mock_flag, test_flag)
        self.skmodel = self._get_skmodel_object(model_name, version_str, start_date, end_date, test_flag)
        self.test_flag = test_flag

    def _set_folder_path(self, version_str, model_name, test_flag):
        self.dict_path = mc.return_base_path(test_flag)
        self.dict_folder = self.dict_path + 'dict/' + version_str + '/'
        self.model_folder = self.dict_path + 'model/' + version_str + '/'
        mu.create_folder(self.dict_folder)
        mu.create_folder(self.model_folder)

    def _set_index_list(self, model_name):
        if model_name == "race":
            self.index_list = ["RACE_KEY", "NENGAPPI"]
        elif model_name == "raceuma":
            self.index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]

    def _set_obj_column_list(self, version_str):
        if version_str == "win":
            self.obj_column_list = ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']
            self.lgbm_params = {
                'WIN_FLAG': {'objective': 'binary'},
                'JIKU_FLAG': {'objective': 'binary'},
                'ANA_FLAG': {'objective': 'binary'},
            }
        elif version_str == "haito":
            self.obj_column_list = ["UMAREN_ARE", "UMATAN_ARE", "SANRENPUKU_ARE"]
            self.lgbm_params = {
                'UMAREN_ARE': {'objective': 'binary'},
                'UMATAN_ARE': {'objective': 'binary'},
                'SANRENPUKU_ARE': {'objective': 'binary'},
            }

    def _set_test_table(self, test_flag):
        """ test用のテーブルをセットする """
        if test_flag:
            self.table_name = self.table_name + "_test"
            self.conn_str = (
                r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                r'DBQ=C:\BaoZ\DB\MasterDB\test_MyDB.MDB;'
            )


    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Load(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _get_skmodel_object(self, model_name, version_str, start_date, end_date, test_flag):
        skmodel = SkModel(model_name, version_str, start_date, end_date, test_flag)
        return skmodel


    def set_learning_df(self):
        base_df = self._get_base_df()
        result_df = self._get_result_df()
        self.learning_df = pd.merge(base_df, result_df, on = self.index_list)
        self.categ_columns = self.skmodel.categ_columns
        self.target_enc_columns = self.skmodel.target_enc_columns

    def create_predict_data(self):
        base_df = self._get_base_df()
        self.categ_columns = self.skmodel.categ_columns
        self.target_enc_columns = self.skmodel.target_enc_columns
        return base_df

    def _get_base_df(self):
        self._set_ld_data()
        base_df = self.skmodel.get_merge_df(self.ld.race_df, self.ld.raceuma_df, self.ld.horse_df, self.ld.prev_raceuma_df, self.ld.grouped_raceuma_prev_df)
        base_df = self.skmodel.get_create_feature_df(base_df)
        base_df = self.skmodel.get_droped_columns_df(base_df)
        base_df = self.skmodel.get_label_encoding_df(base_df)
        base_df = self._rename_key(base_df)
        return base_df

    def _set_ld_data(self):
        """  Loadオブジェクトにデータをセットする処理をまとめたもの。Race,Raceuma,Horse,Prevのデータフレームをセットする

        :param object ld: データロードオブジェクト(ex.LocalBaozLoad)
        """
        self.ld.set_race_df()  # データ取得
        self.ld.set_raceuma_df()
        self.ld.set_horse_df()
        self.ld.set_prev_df(self.ld.race_df, self.ld.raceuma_df)

    def _rename_key(self, df):
        """ キー名を競走コード→RACE_KEY、馬番→UMABANに変更 """
        return_df = df.rename(columns={"競走コード": "RACE_KEY", "馬番": "UMABAN", "月日": "NENGAPPI"})
        return return_df


    def _get_result_df(self):
        self.ld.set_result_df()
        result_df = self.skmodel.get_target_variable_df(self.ld.result_df)
        result_df = self._rename_key(result_df)
        return result_df

    def create_featrue_select_data(self, learning_df):
        self.learning_df = learning_df
        for target_flag in self.obj_column_list:
            print(target_flag)
            self._set_target_flag(target_flag)
            self._create_feature_select_data(target_flag)

    def _set_target_flag(self, target_flag):
        """ 目的変数となるターゲットフラグの値をセットする

        :param str target_flag: ターゲットフラグ名(WIN_FLAG etc.)
        """
        self.target_flag = target_flag

    def _create_feature_select_data(self, target_flag):
        """  指定した説明変数に対しての特徴量作成の処理を行う。TargetEncodingや、Borutaによる特徴選択（DeepLearning以外）を行う

        :param str target_flag:
        """
        print("create_feature_select_data")
        self._set_learning_data(self.learning_df, target_flag)
        self._divide_learning_data()
        self._create_learning_target_encoding()

    def _set_learning_data(self, df, target_column):
        """ 与えられたdfからexp_data,obj_dataを作成する。目的変数が複数ある場合は除外する

        :param dataframe df: dataframe
        :param str target_column: 目的変数のカラム名
        """
        df = df.drop(self.index_list, axis=1)
        self.y_df = df[target_column].copy()
        self.x_df = df.drop(self.obj_column_list, axis=1).copy()
        self._set_label_list(self.x_df)


    def _set_label_list(self, df):
        """ label_listの値にわたされたdataframeのデータ型がobjectのカラムのリストをセットする

        :param dataframe df: dataframe
        """
        self.label_list = df.select_dtypes(include=object).columns.tolist()

    def _divide_learning_data(self):
        """ 学習データをtrainとtestに分ける。オブジェクトのx_df,y_dfからX_train,X_test,y_train,y_testに分けてセットする """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_df, self.y_df, test_size=0.25, random_state=None)

    def _create_learning_target_encoding(self):
        """ TargetEncodeの計算を行い、計算結果のエンコード値をセットする """
        self.X_train = self._set_target_encoding(self.X_train, True, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()

    def _set_target_encoding(self, x_df, fit, y_df):
        """ TargetEncodingの処理をコントロールする処理。label_list（dataframeでデータ型がobjectとなったカラムのリスト）に対してtarget_encodingの処理を行う。
        fitの値がTrueの場合はエンコーダーを作り、Falseの場合は作成されたエンコーダーを適用し、エンコードした値のdataframeを返す。

        :param dataframe x_df: dataframe
        :param bool fit: bool
        :param dataframe y_df: dataframe
        :return: dataframe
        """
        if self.target_enc_columns == "":
            self.target_enc_columns == self.skmodel.target_enc_columns
        target_encoding_columns = list(set(x_df.columns.tolist()) & set(self.target_enc_columns))
        for label in target_encoding_columns:
            x_df.loc[:, "tr_" + label] = self._target_encoding(x_df[label], label, self.target_flag + '_tr_' + label, fit, y_df)
        return x_df

    def _target_encoding(self, sr, label, dict_name, fit, y):
        """ srに対してtarget encodingした結果を返す。fitがTrueの場合はエンコーディングする。ただし、すでに辞書がある場合はそれを使う。

        :param series sr: エンコード対象のSeries
        :param str label: エンコードするラベル名
        :param str dict_name: エンコード辞書名
        :param bool fit: エンコード実施 or 実施済み辞書から変換
        :param series y: 目的変数のSeries
        :return:
        """
        # print("---- target encoding: " + label)
        tr = ce.TargetEncoder(cols=label)
        dict_file = self.dict_folder + '/' + dict_name + '.pkl'
        if fit and not os.path.exists(dict_file):
            tr = tr.fit(sr, y)
            mu.save_dict(tr, dict_name, self.dict_folder)
        else:
            tr = mu.load_dict(dict_name, self.dict_folder)
        sr_tr = tr.transform(sr)
        return sr_tr

    def proc_learning_sk_model(self, df):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
#        if not df.dropna().empty:
        if len(df.index) >= 30:
            print("proc_learning_sk_model: df", df.shape)
            for target in self.obj_column_list:
                print(target)
                self._learning_sk_model(df, target)
        else:
            print("---- 少数レコードのため学習スキップ -- " + str(len(df.index)))
#        else:
#            print("---- NaNデータが含まれているため学習をスキップ")


    def _learning_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str target: str
        """
        this_model_name = self.model_name + "_" + target
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            print("\r\n -- skip create learning model -- \r\n")
        else:
            self._set_target_flag(target)
            print("learning_sk_model: df", df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self._set_learning_data(df, target)
                self._divide_learning_data()
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self._load_learning_target_encoding()
                    self.X_train = self._change_obj_to_int(self.X_train)
                    self.X_test = self._change_obj_to_int(self.X_test)
                    imp_features = self._learning_base_race_lgb(this_model_name, target)
                    # 抽出した説明変数でもう一度Ｌｅａｒｎｉｎｇを実施
                    self.x_df = self.x_df[imp_features]
                    self.categ_columns = list(set(self.categ_columns) & set(imp_features))
                    self.target_enc_columns = list(set(self.target_enc_columns) & set(imp_features))
                    self._divide_learning_data()
                    self._set_label_list(self.x_df) # 項目削除されているから再度ターゲットエンコーディングの対象リストを更新する
                    self._load_learning_target_encoding()
                    self.X_train = self._change_obj_to_int(self.X_train)
                    self.X_test = self._change_obj_to_int(self.X_test)
                    self._learning_race_lgb(this_model_name, target)

    def _load_learning_target_encoding(self):
        """ TargetEncodeを行う。すでに作成済みのエンコーダーから値をセットする  """
        print("load_learning_target_encoding")
        self.X_train = self._set_target_encoding(self.X_train, False, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()


    def _change_obj_to_int(self, df):
        """ objのデータ項目をint型に変更する """
        label_list = df.select_dtypes(include=object).columns.tolist()
        df[label_list] = df[label_list].astype(float) #NaNがあるとintにできない
        return df

    def _learning_base_race_lgb(self, this_model_name, target):
        """ null importanceにより有効な説明変数を抽出する """
        print("learning_base_race_lgb")
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        if self.test_flag:
            num_boost_round = 5
            n_rum = 3
            threshold = 5
            ram_imp_num_boost_round = 5
            early_stopping_rounds = 3
        else:
            num_boost_round = 100
            n_rum = 15
            threshold = 30
            ram_imp_num_boost_round = 100
            early_stopping_rounds = 50

        # データセットを生成する
        # カテゴリ変数を指定する　https://upura.hatenablog.com/entry/2019/10/29/184617
        lgb_train = lgb.Dataset(self.X_train, self.y_train, categorical_feature=self.categ_columns)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train, categorical_feature=self.categ_columns)

        # 上記のパラメータでモデルを学習する
        this_param = self.lgbm_params[target]
        model = lgb_original.train(this_param, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=num_boost_round,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=early_stopping_rounds)

        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X_eval.columns
        imp_df["importance"] = model.feature_importance()
        imp_df = imp_df.sort_values("importance")

        # 比較用のランダム化したモデルを学習する
        null_imp_df = pd.DataFrame()
        for i in range(n_rum):
            print(i)
            ram_lgb_train = lgb.Dataset(self.X_train, np.random.permutation(self.y_train))
            ram_lgb_eval = lgb.Dataset(X_eval, np.random.permutation(y_eval), reference=lgb_train)
            ram_model = lgb_original.train(this_param, ram_lgb_train,
                              # モデルの評価用データを渡す
                              valid_sets=ram_lgb_eval,
                              # 最大で 1000 ラウンドまで学習する
                              num_boost_round=ram_imp_num_boost_round,
                              # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                              early_stopping_rounds=early_stopping_rounds)
            ram_imp_df = pd.DataFrame()
            ram_imp_df["feature"] = X_eval.columns
            ram_imp_df["importance"] = ram_model.feature_importance()
            ram_imp_df = ram_imp_df.sort_values("importance")
            ram_imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, ram_imp_df])

        # 閾値を超える特徴量を取得
        imp_features = []
        for feature in imp_df["feature"]:
            actual_value = imp_df.query(f"feature=='{feature}'")["importance"].values
            null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= threshold:
                imp_features.append(feature)
        print(len(imp_features))

        self._save_learning_model(imp_features, this_model_name + "_feat_columns")
        return imp_features


    def _learning_race_lgb(self, this_model_name, target):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)#, categorical_feature=self.categ_columns)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)#, categorical_feature=self.categ_columns)

        if self.test_flag:
            num_boost_round=5
            early_stopping_rounds = 3
        else:
            num_boost_round=1000
            early_stopping_rounds = 50

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        this_param = self.lgbm_params[target]
        model = lgb.train(this_param, lgb_train,
                          valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)



    def del_set_smote_data(self):
        """ 学習データのSMOTE処理を行い学習データを更新する  """
        # 対象数が少ない場合はサンプリングレートを下げる
        positive_count_train = self.y_train.sum()
        negative_count_train = len(self.y_train) - positive_count_train
        print("check y_train value 0:" + str(negative_count_train) + " 1:" + str(positive_count_train))
        if positive_count_train >= 6:
            smote = BorderlineSMOTE()
            self.X_train, self.y_train = smote.fit_sample(self.X_train, self.y_train)
        else:
            print("----- RandomOverSampler ----- ")
            ros = RandomOverSampler(
                # ratio={1: self.X_train.shape[0], 0: self.X_train.shape[0] // 3}, random_state=71)
                ratio={1: negative_count_train, 0: negative_count_train}, random_state=71)
            # 学習用データに反映
            self.X_train, self.y_train = ros.fit_sample(self.X_train, self.y_train)
        print("-- after sampling: " + str(np.unique(self.y_train, return_counts=True)))

    def _save_learning_model(self, model, model_name):
        """ 渡されたmodelをmodel_nameで保存する。

        :param object model: 学習モデル
        :param str model_name: モデル名
        """
        with open(self.model_folder + model_name + '.pickle', 'wb') as f:
            pickle.dump(model, f)


    def proc_predict_sk_model(self, df):
        """ predictする処理をまとめたもの。指定されたbashoのターゲットフラグ事の予測値を作成して連結したものをdataframeとして返す

        :param dataframe df: dataframe
        :param str val: str
        :return: dataframe
        """
        all_df = pd.DataFrame()
        if not df.empty:
            for target in self.obj_column_list:
                pred_df = self._predict_sk_model(df, target)
                if not pred_df.empty:
                    grouped_df = pred_df  #self._calc_grouped_data(pred_df)
                    grouped_df["target"] = target
                    grouped_df["target_date"] = pred_df["NENGAPPI"].dt.strftime('%Y/%m/%d')
                    grouped_df["model_name"] = self.model_name
                    all_df = pd.concat([all_df, grouped_df]).round(3)
        return all_df

    def _predict_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての予測データの作成を行う

        :param dataframe  df: dataframe
        :param str val: str
        :param str target: str
        :return: dataframe
        """
        self._set_target_flag(target)
        all_pred_df = pd.DataFrame()
        this_model_name = self.model_name + "_" + target
        print("======= this_model_name: " + this_model_name + " ==========")
        with open(self.model_folder + this_model_name + '_feat_columns.pickle', 'rb') as f:
            imp_features = pickle.load(f)
        exp_df = df.drop(self.index_list, axis=1)
        exp_df = exp_df[imp_features]
        exp_df = self._set_predict_target_encoding(exp_df)
        exp_df = exp_df.to_numpy()
        print(exp_df.shape)
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(exp_df)
            pred_df = self._sub_create_pred_df(df, y_pred)
            all_pred_df = pd.concat([all_pred_df, pred_df])
        return all_pred_df

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = temp_df[self.index_list].copy()
        pred_df.loc[:, "prob"] = y_pred
        pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df

    def create_import_data(self, pred_df):
        pred_df.dropna(inplace=True)
        grouped_all_df = pred_df.groupby(["RACE_KEY", "UMABAN", "target"], as_index=False).mean()
        date_df = pred_df[["RACE_KEY", "target_date"]].drop_duplicates()
        temp_grouped_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY")
        grouped = temp_grouped_df.groupby(["RACE_KEY", "target"])
        grouped_df = grouped.describe()['prob'].reset_index()
        merge_df = pd.merge(temp_grouped_df, grouped_df, on=["RACE_KEY", "target"])
        merge_df['predict_std'] = (merge_df['prob'] - merge_df['mean']) / merge_df['std'] * 10 + 50
        temp_grouped_df['predict_rank'] = grouped['prob'].rank("dense", ascending=False)
        merge_df = pd.merge(merge_df, temp_grouped_df[["RACE_KEY", "UMABAN", "predict_rank", "target"]], on=["RACE_KEY", "UMABAN", "target"])
        #return_df = merge_df[['RACE_KEY', 'UMABAN', 'target_date', 'prob', 'predict_std', 'predict_rank']]
        import_df = merge_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        return import_df

    def import_data(self, df):
        """ 計算した予測値のdataframeを地方競馬DBに格納する

        :param dataframe df: dataframe
        """
        cnxn = pyodbc.connect(self.conn_str)
        crsr = cnxn.cursor()
        re_df = df.replace([np.inf, -np.inf], np.nan).dropna()
        date_list = df['target_date'].drop_duplicates()
        for date in date_list:
            print(date)
            target_df = re_df[re_df['target_date'] == date]
            crsr.execute("DELETE FROM " + self.table_name + " WHERE target_date ='" + date + "'")
            crsr.executemany(
                f"INSERT INTO " + self.table_name + " (競走コード, 馬番, 予測フラグ, 予測値, 予測値偏差, 予測値順位, target, target_date) VALUES (?,?,?,?,?,?,?,?)",
                target_df.itertuples(index=False)
            )
            cnxn.commit()

    def create_mydb_table(self):
        """ mydbに予測データを作成する """
        cnxn = pyodbc.connect(self.conn_str)
        create_table_sql = 'CREATE TABLE ' + self.table_name + ' (' \
            '競走コード DOUBLE, 馬番 BYTE, 予測フラグ SINGLE, 予測値 SINGLE, ' \
            '予測値偏差 SINGLE, 予測値順位 BYTE, target VARCHAR(255), target_date VARCHAR(255),' \
            ' PRIMARY KEY(競走コード, 馬番, target));'
        crsr = cnxn.cursor()
        table_list = []
        for talble_info in crsr.tables(tableType='TABLE'):
            table_list.append(talble_info.table_name)
        print(table_list)
        if self.table_name in table_list:
            print("drop table")
            crsr.execute('DROP TABLE ' + self.table_name)
        print(create_table_sql)
        crsr.execute(create_table_sql)
        crsr.commit()
        crsr.close()
        cnxn.close()

    def set_latest_date(self):
        """ 処理済み日の最新日付を取得する """
        cnxn = pyodbc.connect(self.conn_str)
        select_sql = 'SELECT DISTINCT target_date FROM ' + self.table_name + ' WHERE target_date <= #' + self.end_date + '#'
        df = pd.read_sql(select_sql, cnxn)
        max_date = df.max()
        self.start_date = max_date


    def _set_predict_target_encoding(self, df):
        """ 渡されたdataframeに対してTargetEncodeを行いエンコードした値をセットしたdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        self._set_label_list(df)
        df_temp = self._set_target_encoding(df, False, "").copy()
        return df_temp

    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック """
        this_index_list = self.index_list
        this_index_list.remove("NENGAPPI")
        result_df = self._get_result_df()
        check_df = pd.merge(df, result_df, on=this_index_list)
        all_analyze_df = pd.DataFrame()
        for target in self.obj_column_list:
            print(target)
            target_df = check_df[check_df["target"] == target].copy()
            target_df.loc[:, "的中"] = target_df.apply(lambda x: 1 if x[target] == 1 else 0, axis=1)
            prob_analyze_df = target_df.copy()
            prob_analyze_df.loc[:, "bin"] = pd.cut(prob_analyze_df["prob"], 10)
            prob_analyze_df = prob_analyze_df[["bin", "的中", "target"]].copy()
            prob_analyze_df = prob_analyze_df.groupby(["bin", "target"]).describe().reset_index()
            prob_analyze_df.columns = ["bin", "target", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            prob_analyze_df = prob_analyze_df[["bin", "target", "count", "mean"]].copy()
            prob_analyze_df.loc[:, "type"] = "prob"
            all_analyze_df = pd.concat([all_analyze_df, prob_analyze_df])

            std_analyze_df = target_df.copy()
            std_analyze_df.loc[:, "bin"] = pd.cut(std_analyze_df["predict_std"], 10)
            std_analyze_df = std_analyze_df[["bin", "的中", "target"]].copy()
            std_analyze_df = std_analyze_df.groupby(["bin", "target"]).describe().reset_index()
            std_analyze_df.columns = ["bin", "target", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            std_analyze_df = std_analyze_df[["bin", "target", "count", "mean"]].copy()
            std_analyze_df.loc[:, "type"] = "std"
            all_analyze_df = pd.concat([all_analyze_df, std_analyze_df])

        return all_analyze_df