import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)
    correct = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, labels = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            with torch.no_grad():
                output = net(imgs)
                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()

            pbar.update()

    net.train()
    return 100. * correct / 2100
