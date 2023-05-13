import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenetv2
import pytorch_lightning as pl

# 学習済みモデルに合わせた前処理を定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# モデル
# Mobilenet（学習済み）
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = mobilenetv2.mobilenet_v2(pretrained=True)
        self.fc = nn.Linear(1000, 12)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


# 推論
def predict(img):
    # ネットワークの準備
	net = Net().cpu().eval()
	# 学習済みモデルの重み (animal.pt) を読み込み
	net.load_state_dict(torch.load('./animal.pt', map_location=torch.device('cpu')))
	# データの前処理
	img = transform(img)
	img = img.unsqueeze(0)
	# 推論
	y = F.softmax(net(img))
	return y

# 可視化
# Grad-CAMの実装
# def gradcam(img):

