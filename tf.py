python
class YOLOv5:
    def __init__(self, weights):
        self.weights = weights
        self.model = self._build_model()

    def _build_model(self):
        # Load the weights
        model = ...
        return model

    def detect(self, image):
        # Perform object detection on the input image
        detections = ...
        return detections
