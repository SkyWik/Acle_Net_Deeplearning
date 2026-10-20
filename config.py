import torch

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data"
MODEL_SAVE_PATH = "models/acle_net_model.pth"

COGNITIVE_LOAD_CLASSES = ["low_load", "medium_load", "high_load"]
CHART_TYPE_CLASSES = ["bar_chart", "line_chart", "scatter_plot"]
