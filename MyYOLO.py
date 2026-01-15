from ultralytics import YOLO

class MyYOLO:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='classify')
        self.classes = self.model.names

    def predict(self, image):
        results = self.model.predict(
            source=image,   
            imgsz=224,         
            verbose=False,
        )
        return self.classes[results[0].probs.top1],results[0].probs.top1conf.item()