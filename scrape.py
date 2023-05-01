from bs4 import BeautifulSoup
import requests
import json

tokaido = [
    "東京",
    "品川",
    "新横浜",
    "小田原",
    "熱海",
    "三島",
    "新富士（静岡県）",
    "静岡",
    "掛川",
    "浜松",
    "豊橋",
    "三河安城",
    "名古屋",
    "岐阜羽島",
    "米原",
    "京都",
    "新大阪",
]

trains = []

endpoint = "https://www.navitime.co.jp"
url = "https://www.navitime.co.jp/diagram/timetable?node=00006668&lineId=00000110"
response = requests.get(url)
content = response.content

soup = BeautifulSoup(content, "html.parser")
weekday = soup.select_one("#weekday-1 > div")


hours = weekday.select("dl")
for hour in hours:
    trains = hour.select("dd > ul > li")
    for train in trains:
        link = train.select_one("a").get("href")
        res = requests.get(endpoint + link)
        s = BeautifulSoup(res.content, "html.parser")
        train_name = s.select_one("#stoplist-matrix > div > h2").text
        output_data = {
            "direction": "outbound",
            "train_name": train_name,
            "train_type": 1,
            "station_time": [],
        }
        station_time_index = 0

        stations = s.select("#stoplist-matrix > ul > li")
        for station_name in tokaido:
            if (
                station_time_index < len(stations)
                and station_name
                == stations[station_time_index].select("dl > dt")[0].text.strip()
            ):
                station = stations[station_time_index]
                t = station.select("dl > dd")[0].contents

                if station_time_index == 0:
                    arrival_time = None
                    departure_time = t[0].text.strip().split("発")[0]
                elif station_time_index == len(stations) - 1:
                    arrival_time = t[0].text.strip().split("着")[0]
                    departure_time = None
                else:
                    arrival_time = t[0].text.strip().split("着")[0]
                    if len(t) == 1:
                        departure_time = arrival_time
                    else:
                        departure_time = t[2].text.strip().split("発")[0]

                arrival_time = str(
                    int(arrival_time.replace(":", "")) * 100 if arrival_time else ""
                )
                departure_time = str(
                    int(departure_time.replace(":", "")) * 100 if departure_time else ""
                )

                output_data["station_time"].append(
                    {
                        "station_name": station_name,
                        "arrival": arrival_time,
                        "departure": departure_time,
                        "track": 1,
                        "status": "stop",
                    }
                )

                station_time_index += 1
            else:
                output_data["station_time"].append(
                    {
                        "station_name": station_name,
                        "arrival": "",
                        "departure": "",
                        "track": 1,
                        "status": "pass",
                    }
                )

        # JSON ファイルに出力
        file_name = f"trains/{train_name}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
