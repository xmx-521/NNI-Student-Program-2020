import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)
    correct = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, labels = batch['image'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            with torch.no_grad():
                output = net(imgs)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            pbar.update()

    net.train()
    return 100. * correct / n_val
