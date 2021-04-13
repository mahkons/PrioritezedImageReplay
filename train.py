import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm

from samplers import RandomSampler, PrioritizedSampler

BATCH_SIZE = 64
LR = 1e-3
ITERATIONS = 10000
BETA = 1 # TODO BETA anneal ?
PRIORITY_EPS = 0.05
PRIORITY_ALPHA = 1.

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", required=False, choices=["resnet18", "vgg11", "vgg13", "vgg16", "vgg19"])
    parser.add_argument("--sampler", type=str, default="random", required=False, choices=["random", "prioritized_loss"])
    parser.add_argument("--device", type=str, default="cpu", required=False, choices=["cpu", "cuda"])
    return parser


def create_model(model_str):
    return eval("torchvision.models." + model_str + "(pretrained=False, progress=False)")
    
def create_trainsampler(trainset, sampler_str):
    if sampler_str == "random":
        return RandomSampler(len(trainset))
    elif sampler_str == "prioritized_loss":
        return PrioritizedSampler(len(trainset))

def evaluate(model, dataset, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    sumloss = 0.
    correct = 0.
    sum = 0.
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            predicted = torch.argmax(outputs, dim=1)
            sumloss += loss
            correct += (predicted == labels).sum()
            sum += len(images)
    return sumloss / sum, correct / sum


def train(model, trainset, trainsampler, testset, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    sumloss = 0.
    cnt = 0.
    for i in tqdm(range(ITERATIONS)):
        sample_ids, probs = trainsampler.sample(BATCH_SIZE)
        images, labels = zip(*[trainset[sid] for sid in sample_ids])
        images, labels = torch.stack(images).to(device), torch.tensor(labels, device=device)
        probs = torch.tensor(probs, device=device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='none')
        cnt += len(sample_ids)
        sumloss += loss.sum().item()

        weights = (len(trainset) * probs) ** (-BETA)
        # TODO eh?
        #  weights /= weights.max() # would not it introduce bias with small batchsizes?

        # TODO set priority not by loss? for example gradients norm?
        trainsampler.update(sample_ids, (loss + PRIORITY_EPS) ** PRIORITY_ALPHA)

        loss = (loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if i % (ITERATIONS // 100 + 1) == 0:
            model.eval()
            loss, accuracy = evaluate(model, testset, device)
            model.train()
            print("Iteration {}. Train Loss {}. Test Loss {}. Test Accuracy {}".format(i, sumloss / cnt, loss, accuracy))
            cnt = 0.
            sumloss = 0.


if __name__ == "__main__":
    args = create_parser().parse_args()

    transform = T.Compose([T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    device = torch.device(args.device)
    model = create_model(args.model).to(device)
    trainsampler = create_trainsampler(trainset, args.sampler)

    train(model, trainset, trainsampler, testset, device)


