import os
from glob import glob
import json
import requests
from cv2 import imread, IMREAD_COLOR

MY_SUMM_ID = "LRjsmujd76mwUe5ki-cOhcfrpAVsMpsLeA9BZqSl6bMiOI0"

class ChampionData:
    def __init__(self):
        self.latest_patch = self.get_latest_patch()
        self.portraits_path = "data/champion_portraits"
        self.riot_api_key = json.load(open("data/auth.json", encoding="utf-8"))["riotAPIKey"]
        if not os.path.exists(self.get_meta_file_path("champion")):
            self.download_latest_meta_file("champion")
        if not os.path.exists(self.get_meta_file_path("item")):
            self.download_latest_meta_file("item")
        if not os.path.exists(self.get_meta_file_path("summoner")):
            self.download_latest_meta_file("summoner")

    def get_latest_patch(self):
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        try:
            response_json = requests.get(url).json()
            return response_json[0]
        except requests.exceptions.RequestException as exc:
            print("Exception when getting newest game version!")
            print(exc)
            return None

    def get_meta_file_path(self, type):
        return f"data/{type}s-{self.latest_patch}.json"

    def download_latest_meta_file(self, type):
        print(f"Downloading latest {type}s file: '{self.get_meta_file_path(type)}'")
        url = f"http://ddragon.leagueoflegends.com/cdn/{self.latest_patch}/data/en_US/{type}.json"

        old_file = glob("data/champions-*.json")

        try:
            response_json = requests.get(url).json()
            f_out = open(self.get_meta_file_path(type), "w", encoding="utf-8")
            json.dump(response_json, f_out)
            if old_file != []:
                os.remove(old_file[0])
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting newest {type}.json file!")
            print(exc)

    def download_champion_portrait(self, champ_name):
        url = f"http://ddragon.leagueoflegends.com/cdn/{self.latest_patch}/img/champion/{champ_name}.png"
        print(f"Downloading champion portrait for '{champ_name}'")

        try:
            data = requests.get(url, stream=True)
            filename = f"{self.portraits_path}/{champ_name}.png"
            with open(filename, "wb") as fp:
                for chunk in data.iter_content(chunk_size=128):
                    fp.write(chunk)
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting champion portrait for {champ_name}!")
            print(exc)

    def get_champion_names(self):
        with open(self.get_meta_file_path("champion"), encoding="utf-8") as fp:
            champ_data = json.load(fp)
            names = []
            for champ_name in champ_data["data"]:
                names.append(champ_name)
            return names

    def get_champ_name(self, champ_id):
        with open(self.get_meta_file_path("champion"), encoding="utf-8") as fp:
            champ_data = json.load(fp)
            for champ_name in champ_data["data"]:
                if int(champ_data["data"][champ_name]["key"]) == champ_id:
                    return champ_name
        return None

    def get_active_game_data(self):
        api_route = "https://euw1.api.riotgames.com"
        token_header = {"X-Riot-Token": self.riot_api_key}
        try:
            response = requests.get(
                f"{api_route}/lol/spectator/v4/active-games/by-summoner/{MY_SUMM_ID}",
                headers=token_header
            )
            return response.json()
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting active game from Riot API!")
            print(exc)

    def download_champion_portraits(self):
        champ_names = self.get_champion_names()
        champ_portrait_files = glob(f"{self.portraits_path}/*.png")
        portrait_names = [
            x.replace("\\", "/").split("/")[-1].split(".")[0]
            for x in champ_portrait_files
        ]
        for champ_name in champ_names:
            if champ_name not in portrait_names:
                self.download_champion_portrait(champ_name)

    def get_champion_portrait(self, champ_name):
        return imread(f"{self.portraits_path}/{champ_name}.png", IMREAD_COLOR)
