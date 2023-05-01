import glob
import json
from datetime import datetime

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


train_data_list = []

# "trains/{train_name}.json" というパターンに一致するファイル名のリストを取得
file_list = glob.glob("trains/*.json")

# ファイルリストをループして、すべてのファイルを読み込み、JSON をパースし、リストに追加
for file_name in file_list:
    with open(file_name, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        train_data_list.append(train_data)

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
            pass
        # 停車駅の場合
        else:
            if station["arrival"] != "":
                x_data.append(datetime.strptime(station["arrival"], "%H%M%S"))
                y_data.append(station_indices[i])
            if station["departure"] != "":
                x_data.append(datetime.strptime(station["departure"], "%H%M%S"))
                y_data.append(station_indices[i])

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
    p.circle(
        x_data,
        y_data,
        size=5,
        fill_color=train_color,
        line_color=train_color,
        alpha=0.5,
    )


# 出力ファイルの設定
output_file("line_plot.html")


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
