from PIL import Image
import torchvision.transforms as transforms

# Custom crop transformation
class CenterCropLongDimension(object):
    def __init__(self, ):
        pass
    def __call__(self, img):
        width, height = img.size
        if width > height:
            # Crop 80 pixels from each side of the width
            left = 80
            right = width - 80
            top = 0
            bottom = height
        else:
            # Crop 80 pixels from each side of the height
            left = 0
            right = width
            top = 80
            bottom = height - 80
        img = img.crop((left, top, right, bottom))
        return img

    def __repr__(self):
        return "Custom Transform: Crop"