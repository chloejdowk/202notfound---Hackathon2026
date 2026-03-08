# EEG_CNN_PyBullet_Full.py
import os
import time
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pybullet as p
import pybullet_data

# -----------------------------
# PARAMETERS
# -----------------------------
TMIN = 0
TMAX = 4
CHANNELS_KEYWORDS = ['C3..','C4..','Cz..']
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_PATH = "./TRAIN"
TEST_PATH = "./TEST"
MODEL_PATH = "./models/eeg_cnn.pth"

# -----------------------------
# DATA LOADING
# -----------------------------
def load_subject_folder(subject_folder):
    X = []
    y = []
    files = sorted(os.listdir(subject_folder))
    for f in files:
        if not f.lower().endswith(".edf"):
            continue
        file_path = os.path.join(subject_folder, f)
        print(f"Loading {file_path}")
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Could not read {file_path}, skipping. Reason: {e}")
            continue

        # pick channels
        picks = [ch for ch in raw.ch_names if any(k in ch for k in CHANNELS_KEYWORDS)]
        if not picks:
            continue
        raw.pick_channels(picks)

        # get events
        try:
            events, event_dict = mne.events_from_annotations(raw)
        except Exception as e:
            continue

        # only T1/T2 (left/right hand)
        motor_events = {k:event_dict[k] for k in ["T1","T2"] if k in event_dict}
        if not motor_events:
            continue

        epochs = mne.Epochs(raw, events, event_id=motor_events,
                            tmin=TMIN, tmax=TMAX, baseline=None,
                            preload=True, verbose=False)
        data = epochs.get_data()
        labels = epochs.events[:,2]
        labels = labels - labels.min()
        X.append(data)
        y.append(labels)

    if not X:
        return None, None
    return np.concatenate(X), np.concatenate(y)

def load_all_subjects(data_path):
    X_total = []
    y_total = []
    subjects = sorted(os.listdir(data_path))
    for s in subjects:
        subject_folder = os.path.join(data_path, s)
        if not os.path.isdir(subject_folder):
            continue
        print("Processing subject:", s)
        X, y = load_subject_folder(subject_folder)
        if X is not None:
            X_total.append(X)
            y_total.append(y)
    if not X_total:
        raise ValueError("No usable EEG data found.")
    X_total = np.concatenate(X_total)
    y_total = np.concatenate(y_total)
    print("Total epochs loaded:", X_total.shape)
    return X_total, y_total

# -----------------------------
# CNN MODEL
# -----------------------------
class EEG_CNN(nn.Module):
    def __init__(self, channels, samples):
        super().__init__()
        self.conv1 = nn.Conv1d(channels,16,kernel_size=5,padding=2)
        self.conv2 = nn.Conv1d(16,32,kernel_size=5,padding=2)
        self.fc1 = nn.Linear(32*samples,64)
        self.fc2 = nn.Linear(64,2)  # 2 classes: left/right

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for e in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {e+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 2D PYBULLET SIMULATION
# -----------------------------
def run_pybullet_2d(model, X_test):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0,0,-9.8)
    planeId = p.loadURDF("plane.urdf")

    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1,0,0,1])
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
    start_pos = [0,0,0.1]
    ballId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=start_pos)

    p.setRealTimeSimulation(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    vel_x, vel_y = 0.0, 0.0
    step_size = 0.05
    decay = 0.1

    for i in range(len(X_test)):
        trial = X_test_tensor[i].unsqueeze(0)
        with torch.no_grad():
            pred = model(trial)
            action = torch.argmax(pred, dim=1).item()

        # map action to X/Y
        if action == 0:  # left
            vel_x = -step_size
            vel_y = 0
        elif action == 1:  # right
            vel_x = step_size
            vel_y = 0

        pos, orn = p.getBasePositionAndOrientation(ballId)

        # restore toward origin
        vel_x -= decay*pos[0]
        vel_y -= decay*pos[1]

        new_pos = [pos[0]+vel_x, pos[1]+vel_y, pos[2]]
        p.resetBasePositionAndOrientation(ballId,new_pos,orn)
        time.sleep(0.05)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_all_subjects(TRAIN_PATH)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    model = EEG_CNN(n_channels, n_samples)

    # Ensure models folder exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Train or load model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Loaded existing model, skipping training")
    else:
        print("Training CNN...")
        train_model(model, train_loader, EPOCHS, LEARNING_RATE)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model trained and saved to {MODEL_PATH}")

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_all_subjects(TEST_PATH)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Run 2D PyBullet simulation
    print("Running PyBullet 2D simulation...")
    run_pybullet_2d(model, X_test)