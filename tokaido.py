# 列車名から列車種別に基づく色を取得
def get_train_color(train_name):
    if "ひかり" in train_name:
        return "red"
    elif "こだま" in train_name:
        return "blue"
    else:
        return "#FFA500"
