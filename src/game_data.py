import os
from glob import glob
import json
import requests
from cv2 import imread, IMREAD_COLOR

MY_SUMM_ID = "LRjsmujd76mwUe5ki-cOhcfrpAVsMpsLeA9BZqSl6bMiOI0"

class GameData:
    def __init__(self):
        self.latest_patch = self.get_latest_patch()
        self.champ_portraits_path = "data/champion_portraits"
        self.item_icons_path = "data/item_icons"
        self.riot_api_key = json.load(open("data/auth.json", encoding="utf-8"))["riotAPIKey"]
        if not os.path.exists(self.get_meta_file_path("champion")):
            self.download_latest_meta_file("champion")
            self.download_champion_portraits()
        if not os.path.exists(self.get_meta_file_path("item")):
            self.download_latest_meta_file("item")
            self.download_item_icons()
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

    def get_meta_file_path(self, data_type):
        return f"data/{data_type}s-{self.latest_patch}.json"

    def download_latest_meta_file(self, data_type):
        print(f"Downloading latest {data_type}s file: '{self.get_meta_file_path(data_type)}'")
        url = f"http://ddragon.leagueoflegends.com/cdn/{self.latest_patch}/data/en_US/{data_type}.json"

        old_file = glob(f"data/{data_type}s-*.json")

        try:
            response_json = requests.get(url).json()
            f_out = open(self.get_meta_file_path(data_type), "w", encoding="utf-8")
            json.dump(response_json, f_out)
            if old_file != []:
                os.remove(old_file[0])
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting newest {data_type}.json file!")
            print(exc)

    def download_champion_portrait(self, champ_name):
        url = f"http://ddragon.leagueoflegends.com/cdn/{self.latest_patch}/img/champion/{champ_name}.png"
        print(f"Downloading champion portrait for '{champ_name}'")

        try:
            data = requests.get(url, stream=True)
            filename = f"{self.champ_portraits_path}/{champ_name}.png"
            with open(filename, "wb") as fp:
                for chunk in data.iter_content(chunk_size=128):
                    fp.write(chunk)
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting champion portrait for {champ_name}!")
            print(exc)

    def download_item_icon(self, item_id):
        url = f"http://ddragon.leagueoflegends.com/cdn/{self.latest_patch}/img/item/{item_id}.png"
        print(f"Downloading item icon for '{item_id}'")

        try:
            data = requests.get(url, stream=True)
            filename = f"{self.item_icons_path}/{item_id}.png"
            with open(filename, "wb") as fp:
                for chunk in data.iter_content(chunk_size=128):
                    fp.write(chunk)
        except requests.exceptions.RequestException as exc:
            print(f"Exception when getting champion portrait for {item_id}!")
            print(exc)

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

    def download_champion_portraits(self):
        champ_names = self.get_champion_names()
        champ_portrait_files = glob(f"{self.champ_portraits_path}/*.png")
        portrait_names = [
            x.replace("\\", "/").split("/")[-1].split(".")[0]
            for x in champ_portrait_files
        ]
        for champ_name in champ_names:
            if champ_name not in portrait_names:
                self.download_champion_portrait(champ_name)

    def get_champion_portrait(self, champ_name):
        return imread(f"{self.champ_portraits_path}/{champ_name}.png", IMREAD_COLOR)

    def get_item_ids(self):
        with open(self.get_meta_file_path("item"), encoding="utf-8") as fp:
            item_data = json.load(fp)
            return list(item_data["data"].keys())

    def get_item_name(self, item_id):
        if item_id == -1: # Lack of any item.
            return "Nothing"

        with open(self.get_meta_file_path("item"), encoding="utf-8") as fp:
            item_data = json.load(fp)
            for i_id in item_data["data"]:
                if i_id == item_id:
                    return item_data["data"][i_id]["name"]
        return None

    def download_item_icons(self):
        item_ids = self.get_item_ids()
        item_icon_files = glob(f"{self.item_icons_path}/*.png")
        icon_names = [
            x.replace("\\", "/").split("/")[-1].split(".")[0]
            for x in item_icon_files
        ]
        for item_id in item_ids:
            if item_id not in icon_names:
                self.download_item_icon(item_id)

    def get_item_icon(self, icon_id):
        return imread(f"{self.item_icons_path}/{icon_id}.png", IMREAD_COLOR)

    def get_no_item_icon(self):
        return imread("data/empty_icon.png", IMREAD_COLOR)
