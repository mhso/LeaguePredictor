import requests
import json
from game.transponder import send_status

if __name__ == "__main__":
    MY_SUMM_ID = "LRjsmujd76mwUe5ki-cOhcfrpAVsMpsLeA9BZqSl6bMiOI0"
    auth = json.load(open("data/auth.json", encoding="utf-8"))
    pct_win = "80.12"
    transpoder_data = {
        "secret": auth["discordToken"], "pct_win": pct_win,
        "summoner_id": MY_SUMM_ID
    }
    send_status(transpoder_data)

    # predict_url = f"http://mhooge.com:5000/intfar/prediction?game_id=123"
    # try:
    #     predict_response = requests.get(predict_url)
    #     if predict_response.ok:
    #         pct_win = predict_response.json()["response"]
    #         print(f"\nPredicted chance of winning: **{pct_win}%**")
    #     else:
    #         error_msg = predict_response.json()["response"]
    #         print(f"Get game prediction error: {error_msg}")

    # except requests.exceptions.RequestException as e:
    #     print("Exception ignored in !game: " + str(e))
