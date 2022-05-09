# scivision_huggingface

Model repository for the [scivision](https://scivision.readthedocs.io/) project that enables loading of image classification models from [Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads).

Via the scivision API, the [top 10 downloaded Image Classification models from Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads) (of models with a model card, last updated 9th May 2022) can be installed, loaded and run. The list of models is as follows:

1. [microsoft_swin_tiny_patch4_window7_224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
2. [microsoft_beit_base_patch16_224](https://huggingface.co/microsoft/beit-base-patch16-224)
3. [google_vit_base_patch16_224](https://huggingface.co/google/vit-base-patch16-224)
4. [microsoft_beit_base_patch16_224_pt22k_ft22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)
5. [facebook_deit_base_distilled_patch16_224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
6. [microsoft_swin_large_patch4_window7_224](https://huggingface.co/microsoft/swin-large-patch4-window7-224)
7. [google_vit_base_patch32_384](https://huggingface.co/google/vit-base-patch32-384)
8. [nvidia_mit_b0](https://huggingface.co/nvidia/mit-b0)
9. [microsoft_swin_base_patch4_window7_224](https://huggingface.co/microsoft/swin-base-patch4-window7-224)
10. [microsoft_swin_small_patch4_window7_224](https://huggingface.co/microsoft/swin-small-patch4-window7-224)

and a bonus model I found just for fun...

Bonus: [imjeffhi_pokemon_classifier](https://huggingface.co/imjeffhi/pokemon_classifier)

Models in this list can be loaded and used on data with a few lines of code, e.g.

```python
from scivision import load_pretrained_model
this_repo = 'https://github.com/alan-turing-institute/scivision_huggingface'
model = load_pretrained_model(this_repo, allow_install=True, model='microsoft_swin_tiny_patch4_window7_224')
```

You can then use the loaded model's predict function on image data loaded via *scivision* (see the [user guide](https://scivision.readthedocs.io/en/latest/user_guide.html) for details on how data is loaded via the scivision catalog):

```python
model.predict(<image data>)
```