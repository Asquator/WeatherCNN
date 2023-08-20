import torch
DEFAULT_IMG_SIZE = (260, 400)
DATASET_PATH = 'dataset_weather'
TARGET_CATEGORIES = ['fogsmog', 'rain', 'snow']  # 'rime', 'lightning', 'sandstorm', 'shine', 'sunrise', 'cloudy'
MEAN=[0.5324, 0.5285, 0.5349]
STD=[0.2546, 0.2426, 0.2710]
device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
