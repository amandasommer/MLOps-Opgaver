from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

# assume we have a trained model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
preds, target = [], []
for batch in train_dataloader:
    x, y = batch
    probs = model(x)
    preds.append(probs.argmax(dim=-1))
    target.append(y.detach())

target = torch.cat(target, dim=0)
preds = torch.cat(preds, dim=0)

report = classification_report(target, preds)
with open("classification_report.txt", 'w') as outfile:
    outfile.write(report)
confmat = confusion_matrix(target, preds)
disp = ConfusionMatrixDisplay(cm = confmat, )
plt.savefig('confusion_matrix.png')

