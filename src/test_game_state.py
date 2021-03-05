import cv2
from game.game_data import GameData, MY_SUMM_ID
from game.game_state import GameState
from classifier.digit_classifier import DigitClassifier

class TestGameData(GameData):
    def __init__(self, test_image_index):
        super().__init__()
        self.valid_ids = [1, 4, 5, 7, 8, 9]
        if test_image_index not in self.valid_ids:
            print("Invalid image index.")
            exit(0)
        self.champions = []
        self.summs = [
            (12, 4), (4, 11), (4, 14), (4, 7), (4, 14),
            (4, 12), (4, 11), (4, 12), (3, 4), (14, 4)
        ]
        self.summ_ids = [f"Dude {index+1}" for index in range(10)]
        if test_image_index == 1:
            self.champions = [85, 234, 777, 21, 412, 875, 102, 38, 360, 53]
            self.summ_ids[2] = MY_SUMM_ID
        elif test_image_index == 4:
            self.champions = [122, 517, 142, 51, 412, 111, 104, 7, 360, 63]
            self.summ_ids[7] = MY_SUMM_ID
        elif test_image_index == 5:
            self.champions = [223, 141, 4, 360, 432, 98, 11, 236, 53, 61]
            self.summ_ids[3] = MY_SUMM_ID
        elif test_image_index == 7:
            self.champions = [106, 69, 54, 110, 350, 8, 63, 81, 104, 89]
            self.summs = [
                (4, 6), (4, 1), (14, 4), (4, 3), (4, 3),
                (21, 4), (21, 4), (14, 4), (4, 21), (14, 4)
            ]
            self.summ_ids[1] = MY_SUMM_ID
        elif test_image_index == 8:
            self.champions = [34, 23, 10, 21, 19, 223, 876, 62, 235, 86]
            self.summs = [
                (4, 12), (4, 3), (4, 3), (1, 4), (4, 6),
                (4, 6), (4, 7), (14, 4), (4, 6), (4, 3)
            ]
            self.summ_ids[6] = MY_SUMM_ID
        elif test_image_index == 9:
            self.champions = [24, 876, 777, 147, 99, 86, 60, 39, 51, 555]
            self.summs = [
                (21, 4), (4, 11), (4, 14), (4, 3), (4, 14),
                (4, 21), (4, 11), (14, 4), (4, 7), (14, 4)
            ]
            self.summ_ids[2] = MY_SUMM_ID

    def get_active_game_data(self):
        return {
            "participants": [
                {
                    "championId": self.champions[0], "teamId": 100, "summonerId": self.summ_ids[0],
                    "spell1Id": self.get_summoner_spell_index(self.summs[0][0]), # Teleport
                    "spell2Id": self.get_summoner_spell_index(self.summs[0][1]) # Flash
                },
                {
                    "championId": self.champions[1], "teamId": 100, "summonerId": self.summ_ids[1],
                    "spell1Id": self.get_summoner_spell_index(self.summs[1][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[1][1]) # Smite
                },
                {
                    "championId": self.champions[2], "teamId": 100, "summonerId": self.summ_ids[2],
                    "spell1Id": self.get_summoner_spell_index(self.summs[2][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[2][1]) # Ignite
                },
                {
                    "championId": self.champions[3], "teamId": 100, "summonerId": self.summ_ids[3],
                    "spell1Id": self.get_summoner_spell_index(self.summs[3][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[3][1]) # Heal
                },
                {
                    "championId": self.champions[4], "teamId": 100, "summonerId": self.summ_ids[4],
                    "spell1Id": self.get_summoner_spell_index(self.summs[4][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[4][1]) # Ignite
                },
                {
                    "championId": self.champions[5], "teamId": 200, "summonerId": self.summ_ids[5],
                    "spell1Id": self.get_summoner_spell_index(self.summs[5][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[5][1]) # Teleport
                },
                {
                    "championId": self.champions[6], "teamId": 200, "summonerId": self.summ_ids[6],
                    "spell1Id": self.get_summoner_spell_index(self.summs[6][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[6][1]) # Smite
                },
                {
                    "championId": self.champions[7], "teamId": 200, "summonerId": self.summ_ids[7],
                    "spell1Id": self.get_summoner_spell_index(self.summs[7][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[7][1]) # Teleport
                },
                {
                    "championId": self.champions[8], "teamId": 200, "summonerId": self.summ_ids[8],
                    "spell1Id": self.get_summoner_spell_index(self.summs[8][0]), # Exhaust
                    "spell2Id": self.get_summoner_spell_index(self.summs[8][1]) # Flash
                },
                {
                    "championId": self.champions[9], "teamId": 200, "summonerId": self.summ_ids[9],
                    "spell1Id": self.get_summoner_spell_index(self.summs[9][0]), # Ignite
                    "spell2Id": self.get_summoner_spell_index(self.summs[9][1]) # Flash
                } 
            ]
        }

if __name__ == "__main__":
    test_img_index = 4
    img = cv2.imread(f"test_data/frame_{test_img_index}.png", cv2.IMREAD_COLOR)
    champion_data = TestGameData(test_img_index)
    digit_classifier = DigitClassifier()
    digit_classifier.load()
    game_state_handler = GameState(champion_data, digit_classifier)
    state, data = game_state_handler.get_game_state(img)
    game_data, my_team = data
    for team in game_data:
        print(f"====== {team.upper()} TEAM ======")
        print(f"Towers destroyed: {game_data[team]['towers_destroyed']}")
        print(f"Dragons: {game_data[team]['dragons']}")
        for player_data in game_data[team]["players"]:
            print(player_data)
            print("***********************************************")
    
