from movi_models import MoViNet
from movi_config import _C

class Movinet():
	def __init__(self, device):
		model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True)

	def shift(self, pts):
		pts_min = torch.reshape(torch.amin(pts, dim=(1, 2, 3)), (pts.shape[0], 1, 1, 1, pts.shape[4]))
		pts_max = torch.reshape(torch.amax(pts, axis=(1, 2, 3)), (pts.shape[0], 1, 1, 1, pts.shape[4]))
		pts_norm = (pts + pts_min) / (pts_max - pts_min)
		pts = (pts+ pts_min)/(pts_max-pts_min)
		return pts

	def __call__(self, x):
		return self.model(self.shift(x))

