import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, SwinForImageClassification


def tidy_predict(self, image: np.ndarray) -> str:
    """Gives the top prediction for the provided image"""
    pillow_image = Image.fromarray(image.to_numpy(), 'RGB')
    inputs = self.feature_extractor(images=pillow_image, return_tensors="pt")
    outputs = self.pretrained_model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return "Predicted class: " + self.pretrained_model.config.id2label[predicted_class_idx]
        

class microsoft_swin_tiny_patch4_window7_224:
    def __init__(self):
        self.model_name = 'microsoft/swin-tiny-patch4-window7-224'
        self.pretrained_model = SwinForImageClassification.from_pretrained(self.model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
# class google_vit_base_patch16_224:
#     def __init__(self):
#         self.pretrained_model, self.preprocess_input = model_build('google/vit-base-patch16-224')
# 
#     def predict(self, image: np.ndarray) -> np.ndarray:
#         return tidy_predict(self, image)


if __name__ == "__main__":
    pass