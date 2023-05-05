import glob
import json
from datetime import datetime, timedelta
import numpy as np

from bokeh.models import CustomJSTickFormatter, FixedTicker
from bokeh.plotting import figure, output_file, show

from tokaido import get_train_color

# データの読み込み
with open("data.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 駅名のリストを作成
station_names = [station["name"] for station in json_data["stations"]]

# 駅名に対応する目盛り値を作成
station_indices = [station["distance"] for station in json_data["stations"]]

# プロットするデータ
p = figure(
    y_range=(max(station_indices), min(station_indices)),
    x_axis_type="datetime",
    width=2400,
    height=1200,
)


def load_train_data_from_files():
    """
    JSON ファイルから列車データを読み込む

    Returns:
        list: 列車データのリスト
    """

    train_data_list = []

    # "trains/{train_name}.json" というパターンに一致するファイル名のリストを取得
    file_list = glob.glob("trains/*.json")

    # ファイルリストをループして、すべてのファイルを読み込み、JSON をパースし、リストに追加
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            train_data_list.append(train_data)

    return train_data_list


train_data_list = load_train_data_from_files()


BASE_DATETIME = datetime(1900, 1, 1)


def calculate_passing_times(time_data, station_indices):
    """
    通過駅の推定通過時刻を計算し、対応する time_data の位置に追加する関数。

    引数:
        time_data (list): 各駅の時刻データ (通過駅の場合は None) を格納したリスト。
        station_indices (list): 各駅のインデックスデータを格納したリスト。

    戻り値:
        なし。time_data が更新され、通過駅の推定通過時刻が追加されます。
    """

    # 通過駅の推定通過時刻を計算

    # まず、通過駅ではない駅の時刻データをUNIXタイムスタンプに変換し、time_data_np として保存します。
    time_data_np = np.array(
        [(x - BASE_DATETIME).total_seconds() for x in time_data if x is not None]
    )

    # 通過駅ではない駅のインデックスを station_indices_np として保存します。
    station_indices_np = np.array(
        [y for x, y in zip(time_data, station_indices) if x is not None]
    )

    # time_data から通過駅の位置とインデックスを取得し、passing_stations に保存します。
    passing_stations = [
        (i, idx)
        for i, (x, idx) in enumerate(zip(time_data, station_indices))
        if x is None
    ]

    # 通過駅が存在する場合
    if len(passing_stations) > 0:
        # 線形補完を使用して、通過駅の推定通過時刻を計算します。
        # passing_stations のインデックス部分だけを渡して線形補完を行います。
        estimated_passing_times = np.interp(
            [idx for _, idx in passing_stations], station_indices_np, time_data_np
        )

        # 通過駅の推定通過時刻を、対応する time_data の位置に追加します。
        # passing_stations に保存された位置情報 (i) を使用して、time_data に推定通過時刻を追加します。
        for (i, _), passing_time in zip(passing_stations, estimated_passing_times):
            time_data[i] = BASE_DATETIME + timedelta(seconds=passing_time)


x_data_list = []
y_data_list = []

for train in train_data_list:
    x_data = []
    y_data = []

    # 終着駅のインデックスを取得
    terminal_index = next(
        (
            i
            for i, station in enumerate(train["station_time"])
            if station["arrival"] != "" and station["departure"] == ""
        ),
        None,
    )

    # 終着駅が見つからない場合、デフォルト値を設定
    if terminal_index is None:
        terminal_index = len(train["station_time"]) - 1

    for i, station in enumerate(train["station_time"]):
        # 終着駅以降のデータを無視
        if i > terminal_index:
            break

        # 通過駅の場合
        if station["arrival"] == "" and station["departure"] == "":
            x_data.append(None)
            y_data.append(station_indices[i])
        # 停車駅の場合
        else:
            if station["arrival"] != "":
                x_data.append(datetime.strptime(station["arrival"], "%H%M%S"))
                y_data.append(station_indices[i])
            if station["departure"] != "":
                x_data.append(datetime.strptime(station["departure"], "%H%M%S"))
                y_data.append(station_indices[i])

    # 通過駅の推定通過時刻を計算
    calculate_passing_times(x_data, y_data)

    x_data_list.append(x_data)
    y_data_list.append(y_data)

for x_data, y_data, train in zip(x_data_list, y_data_list, train_data_list):
    train_color = get_train_color(train["train_name"])

    # プロット
    p.line(x_data, y_data, line_width=2, line_color=train_color, alpha=0.5)
    p.text(
        x_data[0],
        y_data[0],
        text=[train["train_name"]],
        text_font_size="10pt",
        text_color=train_color,
    )
    # p.circle(
    #     x_data,
    #     y_data,
    #     size=5,
    #     fill_color=train_color,
    #     line_color=train_color,
    #     alpha=0.5,
    # )


# 出力ファイルの設定
output_file("index.html")


# y軸の目盛り位置を設定
p.yaxis.ticker = FixedTicker(ticks=station_indices)

# y軸の目盛りラベルを駅名に変更
station_mapping = {str(y): name for y, name in zip(station_indices, station_names)}
p.yaxis.formatter = CustomJSTickFormatter(
    code="""
    const station_mapping = %s;
    return station_mapping[tick.toFixed(1)];
    """
    % json.dumps(station_mapping)
)

show(p)
