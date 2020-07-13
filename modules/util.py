import os
import shutil
import urllib.request
import zipfile
import glob
import numpy as np
import pandas as pd
import pickle
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce
import bisect
from sklearn.preprocessing import LabelEncoder

def scale_df_for_fa(df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, dict_folder):

    mmsc_filename = dict_folder + mmsc_dict_name + '.pkl'
    if os.path.exists(mmsc_filename):
        mmsc = load_dict(mmsc_dict_name, dict_folder)
        stdsc = load_dict(stdsc_dict_name, dict_folder)
    else:
        mmsc = MinMaxScaler()
        stdsc = StandardScaler()
        mmsc.fit(df[mmsc_columns])
        stdsc.fit(df[stdsc_columns])
        save_dict(mmsc, mmsc_dict_name, dict_folder)
        save_dict(stdsc, stdsc_dict_name, dict_folder)
    mmsc_norm = pd.DataFrame(mmsc.transform(df[mmsc_columns]), columns=mmsc_columns)
    stdsc_norm = pd.DataFrame(stdsc.transform(df[stdsc_columns]), columns=stdsc_columns)
    other_df = df.drop(mmsc_columns, axis=1)
    other_df = other_df.drop(stdsc_columns, axis=1)
    norm_df = pd.concat([mmsc_norm, stdsc_norm, other_df], axis=1)
    return norm_df

def save_dict(dict, dict_name, dict_folder):
    """ エンコードした辞書を保存する

    :param dict dict: エンコード辞書
    :param str dict_name: 辞書名
    """
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder)
    with open(dict_folder + dict_name + '.pkl', 'wb') as f:
        print("save dict:" + dict_folder + dict_name)
        pickle.dump(dict, f)

def load_dict(dict_name, dict_folder):
    """ エンコードした辞書を呼び出す

    :param str dict_name: 辞書名
    :return: encodier
    """
    with open(dict_folder + dict_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def onehot_eoncoding(df, oh_columns, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.OneHotEncoder(cols=oh_columns, handle_unknown='impute')
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    df_ce = ce_fit.transform(oh_df)
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce], axis=1)
    return return_df

def hash_eoncoding(df, oh_columns, num, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.HashingEncoder(cols=oh_columns, n_components=num)
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    #print(dir(ce_fit))
    df_ce = ce_fit.transform(oh_df)
    df_ce.columns = [dict_name + '_' + str(x) for x in list(range(num))]
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce.astype(str)], axis=1)
    return return_df



def setup_basic_auth(url, id, pw):
    """ ベーシック認証のあるURLにアクセス

    :param str url:
    """
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(
        realm=None, uri = url, user = id, passwd = pw
    )
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(auth_handler)
    urllib.request.install_opener(opener)


def unzip_file(filename, download_path, archive_path):
    """  ZIPファイルを解凍する。解凍後のZIPファイルはarvhice_pathに移動

    :param str filename:
    :param str download_path:
    :param str archive_path:
    """
    with zipfile.ZipFile(download_path + filename) as existing_zip:
        existing_zip.extractall(download_path)
    shutil.move(download_path + filename, archive_path + filename)


def get_downloaded_list(type, archive_path):
    """ ARCHIVE_FOLDERに存在するTYPEに対してのダウンロード済みリストを取得する

    :param str type:
    :param str archive_path:
    :return:
    """
    os.chdir(archive_path)
    filelist = glob.glob(type + "*")
    return filelist


def get_file_list(filetype, folder_path):
    """ 該当フォルダに存在するTYPE毎のファイル一覧を取得する

    :param str filetype:
    :param str folder_path:
    :return:
    """
    os.chdir(folder_path)
    filelist = glob.glob(filetype + "*")
    return filelist


def get_latest_file(finish_path):
    """ 最新のダウンロードファイルを取得する（KYIで判断）

    :param str finish_path:
    :return:
    """
    os.chdir(finish_path)
    latest_file = glob.glob("KYI*")
    return sorted(latest_file)[-1]


def int_null(str):
    """ 文字列をintにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return int(str)


def float_null(str):
    """ 文字列をfloatにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return float(str)


def int_bataiju_zogen(str):
    """ 文字列の馬体重増減を数字に変換する

    :param str str:
    :return:
    """
    fugo = str[0:1]
    if fugo == "+":
        return int(str[1:3])
    elif fugo == "-":
        return int(str[1:3])*(-1)
    else:
        return 0


def convert_time(str):
    """ 文字列のタイム(ex.1578)を秒数(ex.1178)に変換する

    :param str str:
    :return:
    """
    min = str[0:1]
    if min != ' ':
        return int(min) * 600 + int(str[1:4])
    else:
        return int_null(str[1:4])


def int_haito(str):
    """ 文字列の配当を数字に変換する。空白の場合は0にする

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return 0
    else:
        return int(str)


def get_kaisai_date(filename):
    """ ファイル名から開催年月日を取得する(ex.20181118)

    :param str filename:
    :return:
    """
    return '20' + filename[3:9]


def move_file(file, folder_path):
    """ ファイルをフォルダに移動する

    :param str file:
    :param str folder_path:
    :return:
    """
    shutil.move(file, folder_path)


def escape_create_text(text):
    """ CREATE SQL文の生成時に含まれる記号をエスケープする

    :param str text:
    :return:
    """
    new_text = text.replace('%', '')
    return new_text


def convert_python_to_sql_type(dtype):
    """ Pythonのデータ型(ex.float)からSQL serverのデータ型(ex.real)に変換する

    :param str dtype:
    :return:
    """
    if dtype == 'float64':
        return 'real'
    elif dtype == 'object':
        return 'nvarchar(10)'
    elif dtype == 'int64':
        return 'real'


def convert_date_to_str(date):
    """ yyyy/MM/ddの文字列をyyyyMMddの文字型に変換する

    :param str date: date yyyy/MM/dd
    :return: str yyyyMMdd
    """
    return date.replace('/', '')


def convert_str_to_date(date):
    """ yyyyMMddの文字列をyyyy/MM/ddの文字型に変換する

    :param str date: date yyyyMMdd
    :return: str yyyy/MM/dd
    """
    return date[0:4] + '/' + date[4:6] + '/' + date[6:8]


def check_df(df):
    """ 与えられたdfの値チェック

    :param dataframe df:
    """
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)

    print("------------ データサンプル ----------------------------")
    print(df.iloc[0])
    print(df.shape)

    print("----------- データ統計量確認 ---------------")
    print(df.describe())

    print("----------- Null個数確認 ---------------")
    df_null = df.isnull().sum()
    print(df_null[df_null != 0])

    print("----------- infinity存在確認 ---------------")
    temp_df_inf = df.replace([np.inf, -np.inf], np.nan).isnull().sum()
    df_inf = temp_df_inf - df_null
    print(df_inf[df_inf != 0])

    print("----------- 重複データ確認 ---------------")
    # print(df[df[["RACE_KEY","UMABAN"]].duplicated()].shape)

    print("----------- データ型確認 ---------------")
    print("object型")
    print(df.select_dtypes(include=object).columns)
    # print(df.select_dtypes(include=object).columns.tolist())
    print("int型")
    print(df.select_dtypes(include=int).columns)
    print("float型")
    print(df.select_dtypes(include=float).columns)
    print("datetime型")
    print(df.select_dtypes(include='datetime').columns)


def trans_baken_type(type):
    if type == 1:
        return '単勝　'
    elif type == 2:
        return '複勝　'
    elif type == 3:
        return '枠連　'
    elif type == 4:
        return '枠単　'
    elif type == 5:
        return '馬連　'
    elif type == 6:
        return '馬単　'
    elif type == 7:
        return 'ワイド'
    elif type == 8:
        return '三連複'
    elif type == 9:
        return '三連単'
    elif type == 0:
        return '合計　'

def label_encoding(sr, dict_name, dict_folder):
    """ srに対してlabel encodingした結果を返す

    :param series sr: エンコードしたいSeries
    :param str dict_name: 辞書の名前
    :param str dict_folder: 辞書のフォルダ
    :return: Series
    """
    le = LabelEncoder()
    dict_name = "le_" + dict_name
    filename = dict_folder + dict_name + '.pkl'
    if os.path.exists(filename):
        le = load_dict(dict_name, dict_folder)
    else:
        le = le.fit(sr.astype('str'))
        save_dict(le, dict_name, dict_folder)
    sr = sr.map(lambda s: 'other' if s not in le.classes_ else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, 'other')
    le.classes_ = le_classes
    sr_ce = le.transform(sr.astype('str'))
    return sr_ce


def get_tansho_df(df):
    """ 単勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    tansho_df1 = df[["競走コード", "単勝馬番1", "単勝払戻金1"]]
    tansho_df2 = df[["競走コード", "単勝馬番2", "単勝払戻金2"]]
    tansho_df3 = df[["競走コード", "単勝馬番3", "単勝払戻金3"]]
    df_list = [tansho_df1, tansho_df2, tansho_df3]
    return_df = arrange_return_df(df_list)
    return return_df

def get_fukusho_df(df):
    """ 複勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    fukusho_df1 = df[["競走コード", "複勝馬番1", "複勝払戻金1"]]
    fukusho_df2 = df[["競走コード", "複勝馬番2", "複勝払戻金2"]]
    fukusho_df3 = df[["競走コード", "複勝馬番3", "複勝払戻金3"]]
    fukusho_df4 = df[["競走コード", "複勝馬番4", "複勝払戻金4"]]
    fukusho_df5 = df[["競走コード", "複勝馬番5", "複勝払戻金5"]]
    df_list = [fukusho_df1, fukusho_df2, fukusho_df3, fukusho_df4, fukusho_df5]
    return_df = arrange_return_df(df_list)
    return return_df

def get_wide_df( df):
    """ ワイド配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    wide_df1 = df[["競走コード", "ワイド連番1", "ワイド払戻金1"]]
    wide_df2 = df[["競走コード", "ワイド連番2", "ワイド払戻金2"]]
    wide_df3 = df[["競走コード", "ワイド連番3", "ワイド払戻金3"]]
    wide_df4 = df[["競走コード", "ワイド連番4", "ワイド払戻金4"]]
    wide_df5 = df[["競走コード", "ワイド連番5", "ワイド払戻金5"]]
    wide_df6 = df[["競走コード", "ワイド連番6", "ワイド払戻金6"]]
    wide_df7 = df[["競走コード", "ワイド連番7", "ワイド払戻金7"]]
    df_list = [wide_df1, wide_df2, wide_df3, wide_df4, wide_df5, wide_df6, wide_df7]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_umaren_df(df):
    """ 馬連配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    umaren_df1 = df[["競走コード", "馬連連番1", "馬連払戻金1"]]
    umaren_df2 = df[["競走コード", "馬連連番2", "馬連払戻金2"]]
    umaren_df3 = df[["競走コード", "馬連連番3", "馬連払戻金3"]]
    df_list = [umaren_df1, umaren_df2, umaren_df3]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_umatan_df(df):
    """ 馬単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    umatan_df1 = df[["競走コード", "馬単連番1", "馬単払戻金1"]]
    umatan_df2 = df[["競走コード", "馬単連番2", "馬単払戻金2"]]
    umatan_df3 = df[["競走コード", "馬単連番3", "馬単払戻金3"]]
    umatan_df4 = df[["競走コード", "馬単連番4", "馬単払戻金4"]]
    umatan_df5 = df[["競走コード", "馬単連番5", "馬単払戻金5"]]
    umatan_df6 = df[["競走コード", "馬単連番6", "馬単払戻金6"]]
    df_list = [umatan_df1, umatan_df2, umatan_df3, umatan_df4, umatan_df5, umatan_df6]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_sanrenpuku_df(df):
    """  三連複配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    sanrenpuku1 = df[["競走コード", "三連複連番1", "三連複払戻金1"]]
    sanrenpuku2 = df[["競走コード", "三連複連番2", "三連複払戻金2"]]
    sanrenpuku3 = df[["競走コード", "三連複連番3", "三連複払戻金3"]]
    df_list = [sanrenpuku1, sanrenpuku2, sanrenpuku3]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_sanrentan_df(df):
    """ 三連単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    sanrentan1 = df[["競走コード", "三連単連番1", "三連単払戻金1"]]
    sanrentan2 = df[["競走コード", "三連単連番2", "三連単払戻金2"]]
    sanrentan3 = df[["競走コード", "三連単連番3", "三連単払戻金3"]]
    sanrentan4 = df[["競走コード", "三連単連番4", "三連単払戻金4"]]
    sanrentan5 = df[["競走コード", "三連単連番5", "三連単払戻金5"]]
    sanrentan6 = df[["競走コード", "三連単連番6", "三連単払戻金6"]]
    df_list = [sanrentan1, sanrentan2, sanrentan3, sanrentan4, sanrentan5, sanrentan6]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def arrange_return_df(df_list):
    """ 内部処理用、配当データの列を競走コード、馬番、払戻に統一する

    :param list df_list: dataframeのリスト
    :return: dataframe
    """
    for df in df_list:
        df.columns = ["競走コード", "馬番", "払戻"]
    return_df = pd.concat(df_list)
    temp_return_df = return_df[return_df["払戻"] != 0]
    return temp_return_df

def separate_umaban(x):
    """ 内部処理用。馬番結果の文字をリスト(4,6),(12,1,3)とかに変換する.0,00といった値の場合は０のリストを返す

    :param str x: str
    :return: list
    """
    umaban = str(x)
    if len(umaban) <= 2:
        return [0, 0, 0]
    if len(umaban) % 2 != 0:
        umaban = '0' + umaban
    if len(umaban) == 6:
        list_umaban = [int(umaban[0:2]), int(umaban[2:4]), int(umaban[4:6])]
    else:
        list_umaban = [int(umaban[0:2]), int(umaban[2:4])]
    return list_umaban

def get_haraimodoshi_dict(haraimodoshi_df):
    """ 払戻用のデータを作成する。extオブジェクトから各払戻データを取得して辞書化して返す。

    :return: dict {"tansho_df": tansho_df, "fukusho_df": fukusho_df}
    """
    tansho_df = get_tansho_df(haraimodoshi_df)
    fukusho_df = get_fukusho_df(haraimodoshi_df)
    umaren_df = get_umaren_df(haraimodoshi_df)
    wide_df = get_wide_df(haraimodoshi_df)
    umatan_df = get_umatan_df(haraimodoshi_df)
    sanrenpuku_df = get_sanrenpuku_df(haraimodoshi_df)
    sanrentan_df = get_sanrentan_df(haraimodoshi_df)
    dict_haraimodoshi = {"tansho_df": tansho_df, "fukusho_df": fukusho_df, "umaren_df": umaren_df,
                         "wide_df": wide_df, "umatan_df": umatan_df, "sanrenpuku_df": sanrenpuku_df, "sanrentan_df": sanrentan_df}
    return dict_haraimodoshi

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        pass

def convert_pace(cd):
    dict = {'H': 3, 'M': 2, 'L': 1}
    return dict.get(cd, 0)

def convert_kyoso_joken_code(cd):
    dict = {'4': 3, '5': 3, '8': 4, '9': 4, '10': '4', '15': 5, '16': 5, '701': 1, '702': 2, '703': 2, '999': 6,  '0': 0}
    return dict.get(cd, 0)