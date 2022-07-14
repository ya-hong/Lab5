import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader

import consts
from DataSet import DataSet, collate_fn
from TSAIE import TSAIE
from TextModel import TextModel
from ImageModel import ImageModel
from ResNet import ResNet
import argparse

dataset = DataSet(train=True)
train_indices = []
validate_indices = []
for i in range(len(dataset)):
    if i % 7 == 0:
        validate_indices.append(i)
    else:
        train_indices.append(i)
train = torch.utils.data.Subset(dataset, train_indices)
validation = torch.utils.data.Subset(dataset, validate_indices)
test = DataSet(train=False)


def get_loss_func():
    count = torch.Tensor(consts.TAG_COUNT)
    weight = torch.sqrt((count.sum()) / (3 * count))
    # weight = torch.Tensor([1, 1, 1])
    loss_func = nn.CrossEntropyLoss(weight=weight)
    return loss_func


def save(model, optimizer, epoch=0):
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(check_point, consts.CHECK_POINT_PATH)


def load(model, optimizer=None):
    try:
        checkpoint = torch.load(consts.CHECK_POINT_PATH)
        model.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    except IOError:
        print("无法找到上次的训练结果")
        return 1


def train_model(model, num_epoch, persist, img_only=False, txt_only=False):
    print("*" * 10 + "TRAIN" + "*" * 10)
    model.train()
    dataloader = DataLoader(train, batch_size=100, collate_fn=collate_fn)
    loss_func = get_loss_func()
    if model.is_cuda():
        loss_func = loss_func.cuda()

    start_epoch = 1
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)
    optimizer = torch.optim.AdamW([
        {"params": model.txt_parameters, "lr": 1e-4},
        {"params": model.img_parameters, "lr": 1e-3}
    ])
    # scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_epoch * 0.2,
    #     num_training_steps=num_epoch
    # )

    if persist:
        start_epoch = load(model, optimizer)

    for epoch in range(num_epoch):
        loss_sum = 0
        acc_count = 0
        for X1, X2, Y in dataloader:
            if img_only:
                X2 *= 0
            if txt_only:
                X1 *= 0

            out = model(X1, X2)
            if out.is_cuda:
                Y = Y.cuda()
            loss = loss_func(out, Y)

            loss_sum += loss.item() * len(Y)
            acc_count += (torch.argmax(out, dim=1) == Y).sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        print("epoch: {}, loss: {}, acc: {}".format(
            epoch + start_epoch, loss_sum / len(train), acc_count / len(train)
        ))
        save(model, optimizer, epoch + start_epoch)

    validate_model(model, True, img_only, txt_only)


def validate_model(model, trained=False, img_only=False, txt_only=False):
    print("*" * 10 + "VALIDATE" + "*" * 10)
    if not trained:
        load(model)
    model.eval()
    dataloader = DataLoader(validation, batch_size=100, collate_fn=collate_fn)
    loss_func = get_loss_func()
    if model.is_cuda():
        loss_func = loss_func.cuda()

    loss_sum = 0
    acc_count = 0

    for X1, X2, Y in dataloader:
        if img_only:
            X2 *= 0
        if txt_only:
            X1 *= 0

        with torch.no_grad():
            out = model(X1, X2)
        if out.is_cuda:
            Y = Y.cuda()

        pred = torch.argmax(out, dim=1)

        print("result ", Y)
        print("predict", pred)
        print()

        loss = loss_func(out, Y)
        loss_sum += loss.item() * len(Y)
        acc_count += (pred == Y).sum()

    print("loss: {}, acc: {}".format(
        loss_sum / len(validation), acc_count / len(validation)
    ))


def prediction(model):
    print("*" * 10 + "PREDICT" + "*" * 10)
    load(model)
    model.eval()
    dataloader = DataLoader(test, batch_size=100, collate_fn=collate_fn)
    pred = []
    for X1, X2, _ in dataloader:
        with torch.no_grad():
            out = model(X1, X2)
        res = torch.argmax(out, dim=1)
        pred.append(res)
    pred = torch.cat(pred)
    content = "guid,tag\n"
    for i, item in enumerate(test):
        line = str(item["guid"]) + "," + consts.int2tag[pred[i]]
        content += line + "\n"
    with open("pred.txt", "w") as f:
        f.write(content)


if __name__ == '__main__':
    model = TSAIE()
    # model = TextModel()
    # model = ResNet(3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="是否使用gpu")
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train", help="训练模型")
    parser_train.add_argument("--epoch", type=int, default=15, help="训练轮次")
    parser_train.add_argument("--discard", action="store_true", help="丢弃上次的训练结果")
    parser_train.add_argument("--txt_only", action="store_true", help="只训练文字")
    parser_train.add_argument("--img_only", action="store_true", help="只训练图像")
    parser_train.set_defaults(func=lambda x: train_model(model, x.epoch, not x.discard, x.img_only, x.txt_only))

    parser_validate = subparsers.add_parser("validate", help="使用次选项仅验证模型")
    parser_validate.set_defaults(func=lambda x: validate_model(model))

    parser_predict = subparsers.add_parser("predict", help="生成预测结果")
    parser_predict.set_defaults(func=lambda x: prediction(model))

    args = parser.parse_args()
    if not args.cpu:
        model = model.cuda()
    args.func(args)
