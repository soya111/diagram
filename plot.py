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

# 駅名のリスト
station_names = [station["name"] for station in json_data["stations"]]

# 始発駅からの距離のリスト
station_indices = [station["distance"] for station in json_data["stations"]]

# 辞書作成: 距離から駅名取得、距離から駅インデックス取得
station_indices_dict = dict(zip(station_indices, station_names))
station_indices_to_index_dict = dict(zip(station_indices, range(len(station_indices))))


def load_train_data_from_files(file_pattern):
    train_data_list = []

    # 引数で指定されたパターンに一致するファイル名のリストを取得
    file_list = glob.glob(file_pattern)

    # ファイルリストをループして、すべてのファイルを読み込み、JSON をパースし、リストに追加
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            train_data_list.append(train_data)

    return train_data_list


train_data_list = load_train_data_from_files("trains/*.json")

# 列車名のリスト
train_names = [train["train_name"] for train in train_data_list]

BASE_DATETIME = datetime(1900, 1, 1)


def calculate_passing_times(train_times, train_distances):
    """
    通過駅の推定通過時刻を計算し、対応する train_times の位置に追加する関数。

    引数:
        train_times (list): 各駅の時刻データ (通過駅の場合は None) を格納したリスト。
        train_distances (list): 各駅のインデックスデータを格納したリスト。

    戻り値:
        list: 通過駅の推定通過時刻が追加された新しいリスト。
    """

    # 新しいリストを作成し、train_times の内容をコピー
    updated_train_times = train_times.copy()

    # 通過駅の推定通過時刻を計算

    # まず、通過駅ではない駅の時刻データをUNIXタイムスタンプに変換し、train_times_np として保存します。
    train_times_np = np.array(
        [
            (x - BASE_DATETIME).total_seconds()
            for x in updated_train_times
            if x is not None
        ]
    )

    # 通過駅ではない駅のインデックスを train_distances_np として保存します。
    train_distances_np = np.array(
        [y for x, y in zip(updated_train_times, train_distances) if x is not None]
    )

    # updated_train_times から通過駅の位置とインデックスを取得し、passing_stations に保存します。
    passing_stations = [
        (i, idx)
        for i, (x, idx) in enumerate(zip(updated_train_times, train_distances))
        if x is None
    ]

    # 通過駅が存在する場合
    if len(passing_stations) > 0:
        # 線形補完を使用して、通過駅の推定通過時刻を計算します。
        # passing_stations のインデックス部分だけを渡して線形補完を行います。
        estimated_passing_times = np.interp(
            [idx for _, idx in passing_stations], train_distances_np, train_times_np
        )

        # 通過駅の推定通過時刻を、対応する updated_train_times の位置に追加します。
        # passing_stations に保存された位置情報 (i) を使用して、updated_train_times に推定通過時刻を追加します。
        for (i, _), passing_time in zip(passing_stations, estimated_passing_times):
            updated_train_times[i] = BASE_DATETIME + timedelta(seconds=passing_time)

    return updated_train_times


# 駅到着、駅出発時刻時点での列車の順序を保持するためのリスト
station_train_times = [
    {"arrival": [], "departure": []} for _ in range(len(station_names))
]

# 列車の発着時刻を保存するためのリスト
train_timetable = {
    train_name: [{"station_name": station_name} for station_name in station_names]
    for train_name in train_names
}

# bokehのプロット用のデータを保存するリスト
x_data_list = []
y_data_list = []

for train in train_data_list:
    train_name = train["train_name"]

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
            train_timetable[train_name][i]["status"] = "pass"
            x_data.append(None)
            y_data.append(station_indices[i])
        # 停車駅の場合
        else:
            train_timetable[train_name][i]["status"] = "stop"
            if station["arrival"] != "":
                arrival_time = datetime.strptime(station["arrival"], "%H%M%S")
                x_data.append(arrival_time)
                y_data.append(station_indices[i])
                station_train_times[i]["arrival"].append((arrival_time, train_name))
                train_timetable[train_name][i]["arrival"] = arrival_time

            if station["departure"] != "":
                departure_time = datetime.strptime(station["departure"], "%H%M%S")
                x_data.append(departure_time)
                y_data.append(station_indices[i])
                station_train_times[i]["departure"].append((departure_time, train_name))
                train_timetable[train_name][i]["departure"] = departure_time

    # 通過駅の推定通過時刻を計算
    updated_train_times = calculate_passing_times(x_data, y_data)

    # 通過駅の推定通過時刻を station_train_times に追加
    for i, (updated_time, y) in enumerate(zip(updated_train_times, y_data)):
        if x_data[i] is None and updated_time is not None:
            station_index = station_indices_to_index_dict[y]
            station_train_times[station_index]["arrival"].append(
                (updated_time, train_name)
            )
            train_timetable[train_name][station_index]["arrival"] = updated_time
            station_train_times[station_index]["departure"].append(
                (updated_time, train_name)
            )
            train_timetable[train_name][station_index]["departure"] = updated_time

    x_data_list.append(updated_train_times)
    y_data_list.append(y_data)

# 列車の到着時刻、出発時刻を時刻順にソート
for station_index, station_train_time in enumerate(station_train_times):
    station_train_time["arrival"].sort(key=lambda x: x[0])
    station_train_time["departure"].sort(key=lambda x: x[0])


def compare_train_orders(prev_station_train_time, current_station_train_time):
    prev_departure_order = [
        train_name for _, train_name in prev_station_train_time["departure"]
    ]
    current_arrival_order = [
        train_name for _, train_name in current_station_train_time["arrival"]
    ]

    return (
        prev_departure_order == current_arrival_order,
        prev_departure_order,
        current_arrival_order,
    )


# 駅間の列車の順序を確認
for i in range(len(station_train_times) - 1):
    prev_station_name = station_names[i]
    current_station_name = station_names[i + 1]
    prev_station_train_time = station_train_times[i]
    current_station_train_time = station_train_times[i + 1]

    orders_match, prev_departure_order, current_arrival_order = compare_train_orders(
        prev_station_train_time, current_station_train_time
    )

    # 列車の本数が変わらないことを確認
    assert len(prev_departure_order) == len(
        current_arrival_order
    ), f"駅 {prev_station_name} から 駅 {current_station_name} で列車の本数が変わりました。"

    if not orders_match:
        print(f"駅間での列車の順序が異なります: 駅 {prev_station_name} から 駅 {current_station_name}")
        for prev_train, current_train in zip(
            prev_departure_order, current_arrival_order
        ):
            if prev_train != current_train:
                print(
                    f"列車が入れ替わりました: {prev_station_name}駅出発順序 {prev_train}, {current_station_name}駅到着順序 {current_train}"
                )
    else:
        print(f"駅間での列車の順序は一致しています: 駅 {prev_station_name} から 駅 {current_station_name}")


def create_plot(station_indices, station_names):
    p = figure(
        y_range=(max(station_indices), min(station_indices)),
        x_axis_type="datetime",
        width=2400,
        height=1200,
    )

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

    return p


def plot_train_data(p, x_data, y_data, train_name, train_color):
    p.line(x_data, y_data, line_width=2, line_color=train_color, alpha=0.5)
    p.text(
        x_data[0],
        y_data[0],
        text=[train_name],
        text_font_size="10pt",
        text_color=train_color,
    )


# プロットするデータ
p = create_plot(station_indices, station_names)

for x_data, y_data, train in zip(x_data_list, y_data_list, train_data_list):
    train_color = get_train_color(train["train_name"])
    plot_train_data(p, x_data, y_data, train["train_name"], train_color)


# 出力ファイルの設定
output_file("index.html")

show(p)
