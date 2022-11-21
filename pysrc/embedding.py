from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch
from PIL import Image


def extract_features(filename, model=None):
    if model is None:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    # preprocess the image
    tfms = transforms.Compose([transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]),])

    img = tfms(Image.open(filename)).unsqueeze(0)
    features = model.extract_features(img)

    # avg pooling layer to reduce dimensionality
    pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)

    # reshape to flat 1d array
    pooled = torch.reshape(pooled, (1, 1280))[0]
    return pooled.detach().numpy()



if __name__ == "__main__":
    features = extract_features("../data/test/images/test_2779.JPEG")
    print(features)
    print(features.shape)
