#建模主线
#CSV文件
#-> pandas读入DataFrame
#-> 建立字符词表
#-> 句子转成定长索引序列
#-> Embedding + Flatten + MLP
#-> CrossEntropyLoss
#-> Adam优化
#-> 训练 / 验证 / 测试
#-> matplotlib画曲线
#-> 单句预测

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset


TRAIN_PATH = "emotion_dataset_300_train.csv"
VAL_PATH = "emotion_dataset_300_val.csv"
TEST_PATH = "emotion_dataset_300_test.csv"
#设置初始数值
MAX_LEN = 30
EMBED_DIM = 24
HIDDEN_DIM = 64
NUM_CLASSES = 4
BATCH_SIZE = 30
EPOCHS = 4
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#建立分类表
LABEL_NAMES = {
    0:"开心",
    1:"愤怒",
    2:"伤心",
    3:"中性",
}
#读取数据+清洗
def load_data(path):
    df = pd.read_csv(path,encoding="utf-8-sig")
    df["text"] = df["text"].astype(str).str.strip()#转换成字符串+去空格
    df["label"] = df["label"].astype(int)
    return df
#创建词表函数
def build_vocab(train_df,min_freq=1): #最小词阈值1
    counter = Counter()

    for text in train_df["text"]:
        counter.update(list(text))#把字符串拆成字符+统计

    vocab = {
        "<PAD>":0,
        "<UNK>":1
    }

    for ch,freq in counter.items():
        if freq >= min_freq:
            vocab[ch] = len(vocab)

    return vocab#把每个词给上id

#文本转化成索引
def text_to_ids(text,vocab,max_len):
    ids = [vocab.get(ch,vocab["<UNK>"]) for ch in text]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]]*(max_len-len(ids))
        #这一步是生成一个[0,0,0]拼接在原始后面
    return torch.tensor(ids,dtype=torch.long)

#Dataset--把原始数据变成能用来训练的
class EmotionDataset(Dataset):
    def __init__(self,df,vocab,max_len):
        self.df = df.reset_index(drop=True)#丢弃旧索引
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx,"text"]
        label = int(self.df.loc[idx,"label"])
        #loc按名字找，iloc按位置
        x = text_to_ids(text,self.vocab,self.max_len)
        y = torch.tensor(label,dtype=torch.long)

        return x,y
    #把每个样本按照标准格式取出来

#定义模型
class EmotionMLP(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_len,hidden_dim,num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0#这些点不参与训练
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),#从第一维往后展平
            nn.Linear(max_len*embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,num_classes)
        )

    def forward(self, x):
            x = self.embedding(x)
            logits = self.classifier(x)
            return logits

#定义评估
def evaluate(model,loader,criterion,device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits,y)
            #这里会自动变成平均损失，但我要变成总损失
            total_loss += loss.item()*x.size(0)

            preds = torch.argmax(logits,dim=1)
            total_correct += (preds == y).sum().item()
            total_count += x.size(0)
    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc

#定义训练模型
def train_model(model,train_loader,val_loader,criterion,optimizer,device,epochs):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0

        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()*x.size(0)

            preds = torch.argmax(logits,dim=1)#每行找最大
            total_train_correct += (preds == y).sum().item()
            total_train_count += x.size(0)

        train_loss = total_train_loss / total_train_count
        train_acc = total_train_correct / total_train_count

        val_loss, val_acc = evaluate(model,val_loader,criterion,device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
    return history
#数据绘图

def plot_history(history):
    epochs = range(1,len(history["train_loss"])+1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs,history["train_loss"],label="train_loss")
    plt.plot(epochs,history["val_loss"],label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()#自动调整布局
    plt.show()

#定义预测文本
def predict(model,text,vocab,max_len,device):
    model.eval()

    with torch.no_grad():
        x = text_to_ids(text,vocab,max_len).unsqueeze(0).to(device)
        #加入一个batch维，因为默认有batch维输入
        logits = model(x)
        pred = torch.argmax(logits,dim=1).item()

    return pred,LABEL_NAMES[pred]

#main测试函数

def main():
    train_df = load_data(TRAIN_PATH)
    val_df = load_data(VAL_PATH)
    test_df = load_data(TEST_PATH)

    print("训练集前五行")
    print(train_df.head(),"\n")

    print("训练集类别分布")
    print(train_df["label"].value_counts().sort_index(),"\n")
    #统计每个类别出现次数

    print(f"训练集大小:{len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}\n")

    vocab = build_vocab(train_df)
    print(f"词表大小: {len(vocab)}\n")

    train_dataset = EmotionDataset(train_df,vocab,MAX_LEN)
    val_dataset = EmotionDataset(val_df,vocab,MAX_LEN)
    test_dataset = EmotionDataset(test_df,vocab,MAX_LEN)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model = EmotionMLP(
        vocab_size=len(vocab),
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(model)#打印神经网络结构
    print(f"\n运行设备：{DEVICE}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer_t= torch.optim.Adam(model.parameters(),lr=LR)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_t,
        epochs=EPOCHS,
        device=DEVICE,
    )

    test_loss, test_acc = evaluate(model,test_loader,criterion,DEVICE)
    print(f"\n测试集结果: loss={test_loss:.4f}, acc={test_acc:.4f}")

    plot_history(history)

    torch.save(
        {"model_state_dict": model.state_dict(),
         "vocab": vocab,
         "max_len": MAX_LEN,}
        ,"emotion_mlp_embedding_flatten.pth"
    )
    print("\n模型已保存到emotion_mlp_embedding_flatten.pth")

    demo_texts = [
       ""
    ]

    print("\n=======demo prediction=========")
    for text in demo_texts:
        pred_id,pred_name = predict(model,text,vocab,MAX_LEN,DEVICE)
        print(f"{text} -> 预测类别: {pred_name} ({pred_id})")

if __name__ == "__main__":
    main()
