import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Anime_Dataset
from generator_model import GeneratorUNET
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx ,(x,y) in enumerate(loop):
        x = x.to(config.device)
        y = y.to(config.device)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss ) /2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.l1_lambda
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():
    disc = Discriminator(in_channels=3,feature=64).to(config.device)
    gen = GeneratorUNET(in_channels=3, feature=64).to(config.device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.Load_model:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.learning_rate,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.learning_rate,
        )
    train_dataset =Anime_Dataset(root_dir=config.train_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_worker,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = Anime_Dataset(root_dir=config.val_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.num_epochs):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE, g_scaler, d_scaler,
        )

        if config.Save_model and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()