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
        self.champ_names = {}
        self.champ_handles = {}
        self.champ_index = {}
        self.item_names = {}
        self.item_index = {}
        self.summ_names = {}
        self.summ_index = {}

        if not os.path.exists(self.get_meta_file_path("champion")):
            self.download_latest_meta_file("champion")
            self.download_champion_portraits()
        if not os.path.exists(self.get_meta_file_path("item")):
            self.download_latest_meta_file("item")
            self.download_item_icons()
        if not os.path.exists(self.get_meta_file_path("summoner")):
            self.download_latest_meta_file("summoner")

        self._cache_champ_data()
        self._cache_item_data()
        self._cache_summoners_data()

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

    def get_champion_ids(self):
        return list(self.champ_names.keys())

    def get_champion_names(self):
        return list(self.champ_names.values())

    def get_champion_handles(self):
        return list(self.champ_handles.values())

    def get_champ_name(self, champ_id):
        return self.champ_names.get(champ_id)

    def get_champion_id(self, champ_handle):
        for champ_id in self.champ_handles:
            if self.champ_handles[champ_id] == champ_handle:
                return champ_id
        return None

    def get_champ_handle(self, champ_id):
        return self.champ_handles.get(champ_id)

    def get_champion_index(self, champ_id):
        return self.champ_index[champ_id]

    def get_summoner_spell_ids(self):
        return list(self.summ_names.keys())

    def get_summoner_spell_name(self, summ_id):
        return self.summ_names[summ_id]

    def get_summoner_spell_index(self, summ_id):
        return self.summ_index[summ_id]

    def download_champion_portraits(self):
        champ_names = self.get_champion_handles()
        champ_portrait_files = glob(f"{self.champ_portraits_path}/*.png")
        portrait_names = [
            x.replace("\\", "/").split("/")[-1].split(".")[0]
            for x in champ_portrait_files
        ]
        for champ_name in champ_names:
            if champ_name not in portrait_names:
                self.download_champion_portrait(champ_name)

    def get_champion_portrait(self, champ_name):
        return imread(f"{self.champ_portraits_path}/{champ_name.replace(' ', '')}.png", IMREAD_COLOR)

    def get_item_ids(self):
        return list(self.item_names.keys())

    def get_item_name(self, item_id):
        if item_id == -1: # Lack of any item.
            return self.item_names[0]

        return self.item_names.get(item_id)

    def get_item_index(self, item_id):
        return self.item_index.get(item_id, 0)

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
        if icon_id == 0:
            return self.get_no_item_icon()
        return imread(f"{self.item_icons_path}/{icon_id}.png", IMREAD_COLOR)

    def get_no_item_icon(self):
        return imread("data/empty_icon.png", IMREAD_COLOR)

    def _cache_champ_data(self):
        with open(self.get_meta_file_path("champion"), encoding="utf-8") as fp:
            champ_data = json.load(fp)
            for index, champ_name in enumerate(champ_data["data"]):
                name = champ_data["data"][champ_name]["name"]
                key = int(champ_data["data"][champ_name]["key"])
                self.champ_names[key] = name
                self.champ_handles[key] = champ_name
                self.champ_index[key] = index

    def _cache_item_data(self):
        with open(self.get_meta_file_path("item"), encoding="utf-8") as fp:
            item_data = json.load(fp)
            self.item_names[0] = "Nothing"
            self.item_index[0] = 0
            for index, item_id in enumerate(item_data["data"], start=1):
                item_id_num = int(item_id)
                self.item_names[item_id_num] = item_data["data"][item_id]["name"]
                self.item_index[item_id_num] = index

    def _cache_summoners_data(self):
        with open(self.get_meta_file_path("summoner"), encoding="utf-8") as fp:
            summ_data = json.load(fp)
            for index, summ_name in enumerate(summ_data["data"]):
                key = int(summ_data["data"][summ_name]["key"])
                self.summ_names[key] = summ_data["data"][summ_name]["name"]
                self.summ_index[key] = index
