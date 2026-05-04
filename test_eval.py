import torch
from models.yolo_model import YOLOModel
from utils.evaluation import evaluate_model_on_voc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOModel(device, 'yolo11n.pt')
model.load()

try:
    res = evaluate_model_on_voc(model, num_samples=5)
    print("Evaluation successful:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
