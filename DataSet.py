import torch
from os.path import join
from PIL import Image
from torchvision.transforms import transforms
from torch.nn.utils.rnn import pad_sequence
import consts
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel

img_transform = transforms.Compose([
    transforms.Resize(consts.IMG_SIZE),  # 缩放图片(Image)
    transforms.CenterCrop(consts.IMG_SIZE),  # 从图片中间切出的图片
    transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
])
img_arr = [None]
for i in range(1, 5130):
    try:
        img_path = join(consts.DATA_FOLD, str(i) + ".jpg")
        img = Image.open(img_path)
        img_arr.append(img_transform(img))
    except IOError:
        img_arr.append(None)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(DataSet, self).__init__()

        self.train = train
        self.dataset = []

        train_path = join(consts.FOLD, "train.txt")
        test_path = join(consts.FOLD, "test_without_label.txt")

        path = train_path if train else test_path

        with open(path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue
            (guid, tag) = line.strip().split(",")
            self.dataset.append((int(guid), tag))

        self.item = dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.item:
            (guid, tag) = self.dataset[idx]

            txt_path = join(consts.DATA_FOLD, str(guid) + ".txt")
            img_path = join(consts.DATA_FOLD, str(guid) + ".jpg")

            data = {
                "img": None,
                "txt": None,
                "tag": tag,
                "guid": guid
            }

            with open(txt_path, "rb") as f:
                data["txt"] = str(f.read())

            data["img"] = img_arr[guid]

            self.item[idx] = data
        return self.item[idx]


tokenizer = AutoTokenizer.from_pretrained(consts.MODEL_NAME)


def collate_fn(batch):
    imgs = []
    txts = []
    tags = []
    for item in batch:
        imgs.append(item["img"])

        txt = item["txt"]
        if len(txt) > 510:
            txt = txt[0:128] + txt[-392:]
        txts.append(txt)

        tags.append(consts.tag2int[item["tag"]])

    imgs = pad_sequence(imgs, batch_first=True)
    txts = torch.tensor(tokenizer(txts, padding=True).input_ids)
    tags = torch.tensor(tags).long()
    return imgs, txts, tags
