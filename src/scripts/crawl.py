import json
import requests
from glob import glob
from time import sleep

DATA_PATH = "data/training_data/games"

def get_start_id():
    files = glob(f"{DATA_PATH}/*.json")
    if files == []:
        return 5071853017

    latest = files[-1]
    return int(latest.replace("\\", "/").split("/")[-1].split(".")[0].split("_")[0]) + 1

def crawl_for_game_data(riot_api_key):
    api_route = "https://euw1.api.riotgames.com"
    token_header = {"X-Riot-Token": riot_api_key}
    timeline_endpoint = "lol/match/v4/timelines/by-match"
    match_endpoint = "lol/match/v4/matches"
    start_id = get_start_id()
    legal_queues = set([4, 6, 9, 42, 410, 420, 440, 700])
    endpoints = [("match", match_endpoint), ("timeline", timeline_endpoint)]
    for match_id in range(start_id, 5091939533, 10):
        for desc, endpoint in endpoints:
            try:
                response = requests.get(
                    f"{api_route}/{endpoint}/{match_id}",
                    headers=token_header
                )
                if response.ok:
                    json_data = response.json()
                    if desc == "match":
                        if not json_data["queueId"] in legal_queues:
                            print(f"Skipped game with ID {match_id} (not Ranked SR).", flush=True)
                            sleep(1)
                            break # Skip this match as well as timeline data.

                    json.dump(
                        json_data,
                        open(f"{DATA_PATH}/{match_id}_{desc}.json", "w", encoding="utf-8")
                    )
                    print(f"Saved {desc} data for id {match_id}", flush=True)
                elif response.status_code == 404:
                    print(f"{desc.capitalize()} data for {match_id} does not exist.")
                    sleep(1)
                    break
                else:
                    print(f"Error when saving {desc} data for id {match_id}. Status: {response.status_code}", flush=True)
            except requests.exceptions.RequestException as exc:
                print(f"Exception when getting active game from Riot API! Status: {response.status_code}", flush=True)
                print(exc, flush=True)
            sleep(1)

if __name__ == "__main__":
    api_key = json.load(open("data/auth.json", encoding="utf-8"))["riotAPIKey"]
    crawl_for_game_data(api_key)
