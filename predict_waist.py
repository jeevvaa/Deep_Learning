# predict_waist.py

import torch, torch.nn as nn
import numpy as np, joblib
from sklearn.preprocessing import StandardScaler

# 1. Same model definition
class SimpleNet(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc1 = nn.Linear(D, 64)
        self.act = nn.LeakyReLU(0.1)
        self.out = nn.Linear(64, 1)
    def forward(self, x):
        return self.out(self.act(self.fc1(x)))

# 2. Load scalers & models
scaler_X = joblib.load('waist_model/scaler_X.joblib')
scaler_y = joblib.load('waist_model/scaler_y.joblib')
models   = []
device   = torch.device('cpu')
for f in range(1,6):
    m = SimpleNet(scaler_X.mean_.shape[0])
    state = torch.load(f'waist_model/net_weights_fold{f}.pth', map_location=device)
    m.load_state_dict(state); m.eval()
    models.append(m)

# 3. Feature engineering helper
def fe(x):
    chest, hip, height, weight, gender = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4]
    bmi   = weight / ((height/1000.)**2)
    c2h   = chest / height
    h2h   = hip   / height
    return np.stack([chest,hip,height,weight,gender,bmi,c2h,h2h],axis=1)

def predict(measurements):
    """
    measurements: B×5 numpy array or torch tensor
    columns = [Chest, Hip, Height, Weight, Gender]
    Returns: B×1 torch tensor of Waist (mm)
    """
    arr = measurements.detach().cpu().numpy() if torch.is_tensor(measurements) else np.array(measurements, dtype=np.float32)
    X = fe(arr)
    Xs = scaler_X.transform(X)
    Xt = torch.from_numpy(Xs)
    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m(Xt).cpu().numpy())
    # average across folds
    P = np.mean(np.stack(preds,axis=0),axis=0)
    # invert target scale
    out = scaler_y.inverse_transform(P).astype(np.float32)
    return torch.from_numpy(out)


