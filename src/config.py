from datetime import datetime

CHAR_INPUT_DIM = (3, 32, 32)
GAME_INPUT_DIM = (2, 5, 380)
TRAIN_WITH_GPU = True

LOG_WARNING = 1
LOG_ERROR = 2

def log(data, severity_level=0, end="\n"):
    curr_time = datetime.now()
    prefix = curr_time.strftime("%Y-%m-%d %H:%M:%S")
    if severity_level == LOG_WARNING:
        prefix = prefix + " - [Warning]"
    elif severity_level == LOG_ERROR:
        prefix = prefix + " - [### ERROR ###]"

    print(prefix + " - " + str(data), flush=True, end=end)
