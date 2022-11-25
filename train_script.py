# %%
import wandb
import os 
wandb.init(project="lab3", entity="qwertyforce",reinit=True)

# %%
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import random

_classes={"cloudy":0,"rain":1,"shine":2,"sunrise":3}
_classes_arr = ["cloudy","rain","shine","sunrise"]

dataset_path = "./datasets"
testing_path = "./testing"

folders = os.listdir(dataset_path)
aug_folders = [file for file in folders if "_aug_" in file]
aug_folders_last_aug_ver = {}

for folder in aug_folders:
    prefix = folder[:folder.index("_aug_")]
    aug_num = int(folder[folder.index("_aug_")+5:])
    aug_folders_last_aug_ver[prefix] = max(aug_folders_last_aug_ver.get(prefix,0), aug_num)

training = []
TRAINING_FOLDERS = []

for folder_name, aug_ver in tqdm(aug_folders_last_aug_ver.items()):
    folder_name+=f"_aug_{aug_ver}"
    TRAINING_FOLDERS.append(folder_name)
    for _class in _classes:   #<------------
        for file_name in os.listdir(f"{dataset_path}/{folder_name}/{_class}"):
            training.append((f"{dataset_path}/{folder_name}/{_class}/{file_name}",_classes[_class]))

testing = []
for _class in _classes:   #<------------
    for file_name in os.listdir(f"{testing_path}/{_class}"):
        testing.append((f"{testing_path}/{_class}/{file_name}",_classes[_class]))
random.shuffle(training)

# %%
from PIL import Image
def read_img_file(f):
    img = Image.open(f)
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    return img

# %%
_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# %%
class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx][0]
        _class = self.images[idx][1]
        # print(file_name,_class)
        # img_path = self.IMAGE_PATH+"/"+file_name
        try:
            img = read_img_file(img_path)
            img = _transform(img)
            return (img, _class, img_path)
        except Exception as e:
            print(e)
            print(f"error reading {img_path}")

# %%
train_dataset = CustomDataset(training)
test_dataset = CustomDataset(testing)
NUM_WORKERS = 1

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10

PREFETCH_FACTOR = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# %%
import timm
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.head=torch.nn.Linear(model.head.in_features, 4)
model = model.to(device)

# %%
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = None

# %%
wandb.config.update({
  "learning_rate": LEARNING_RATE,
  "epochs": 10,
  "batch_size": EPOCHS,
  "training_folders": TRAINING_FOLDERS
})

# %%
def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = test_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            # print(phase)
            # exit()
            for inputs, labels,_ in dataloader:
                # print(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            if phase == 'train':
                wandb.log({"loss_train": epoch_loss,"epoch":epoch})
            else:
                wandb.log({"loss_test": epoch_loss,"epoch":epoch})
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

# %%
train_model(model, loss, optimizer, scheduler, num_epochs=EPOCHS)

# %%
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
acc_true=0
acc_all = 0
for inputs, labels, img_path in tqdm(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        preds = model(inputs)
        loss_value = loss(preds, labels)
        preds_class = preds.argmax(dim=1)
        if preds_class == labels.data:
            acc_true+=1
        else:
            print(preds)
            print(img_path)
            print(_classes_arr[preds_class.cpu().numpy()[0]])
            
    acc_all+=1      
wandb.log({"test_acc": acc_true/acc_all})

model_scripted = torch.jit.script(model)
model_scripted.save(f'./{wandb.run.id}_{wandb.run.name}.pt')