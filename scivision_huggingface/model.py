import numpy as np
from PIL import Image
from transformers import (AutoFeatureExtractor,
                          SwinForImageClassification,
                          ViTFeatureExtractor,
                          ViTForImageClassification
                         )


def tidy_predict(self, image: np.ndarray) -> str:
    """Gives the top prediction for the provided image"""
    pillow_image = Image.fromarray(image.to_numpy(), 'RGB')
    inputs = self.feature_extractor(images=pillow_image, return_tensors="pt")
    outputs = self.pretrained_model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return "Predicted class: " + self.pretrained_model.config.id2label[predicted_class_idx]
    
    
def build_vit_model(model_name: str):
    model = ViTForImageClassification.from_pretrained(model_name)
    features = ViTFeatureExtractor.from_pretrained(model_name)
    return model, features
    
    
def build_swin_model(model_name: str):
    model = SwinForImageClassification.from_pretrained(model_name)
    features = AutoFeatureExtractor.from_pretrained(model_name)
    return model, features
        

class microsoft_swin_tiny_patch4_window7_224:
    def __init__(self):
        self.model_name = 'microsoft/swin-tiny-patch4-window7-224'
        self.pretrained_model, self.feature_extractor = build_swin_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class google_vit_base_patch16_224:
    def __init__(self):
        self.model_name = 'google/vit-base-patch16-224'
        self.pretrained_model, self.feature_extractor = build_vit_model(self.model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)
        
        
class imjeffhi_pokemon_classifier:
    def __init__(self):
        self.model_name = 'imjeffhi/pokemon_classifier'
        model, self.feature_extractor = build_vit_model(self.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_model = model.to(device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        return tidy_predict(self, image)


if __name__ == "__main__":
    pass