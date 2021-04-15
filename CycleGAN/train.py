import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Color_enhancement
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import discriminator
from generator_model import Generator

def train(disc_O, disc_A, gen_O, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    O_reals = 0
    O_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (orange, apple) in enumerate(loop):
        orange = orange.to(config.device)
        apple = apple.to(config.device)
        #Train discriminator for I and D
        with torch.cuda.amp.autocast():
            fake_orange = gen_A(apple)
            D_O_real = disc_O(orange)
            D_O_fake = disc_O(fake_orange.detach())
            O_reals += D_O_real.mean().item()
            O_fakes += D_O_fake.mean().item()
            D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
            D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
            D_O_loss = D_O_real_loss + D_O_fake_loss


            fake_apple = gen_O(orange)
            D_A_real = disc_A(apple)
            D_A_fake = disc_A(fake_apple.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss
            D_loss = (D_O_loss + D_A_loss) /2
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_O_fake = disc_O(orange)
            D_A_fake = disc_A(apple)
            loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake))
            loss_G_A = mse(D_A_fake, torch.zeros_like(D_A_fake))

            cycle_orange = gen_A(fake_apple)
            cycle_apple = gen_O(fake_orange)
            cycle_orange_loss = l1(orange, cycle_orange)
            cycle_apple_loss =l1(apple, cycle_apple)

            identity_orange = gen_O(orange)
            identity_apple = gen_A(apple)
            identity_orange_loss = l1(orange, identity_orange)
            identity_apple_loss = l1(apple, identity_apple)

            G_loss = (
                loss_G_O +
                loss_G_A +
                cycle_orange_loss * config.lambda_cycle +
                cycle_apple_loss * config.lambda_cycle +
                identity_orange_loss * config.lambda_identity +
                identity_apple_loss * config.lambda_identity
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_orange * 0.5 + 0.5, f"saved_images/iphone_{idx}.png")
            save_image(fake_apple * 0.5 + 0.5, f"saved_images/dsrl_{idx}.png")

        loop.set_postfix(H_real=O_reals / (idx + 1), H_fake=O_fakes / (idx + 1))

def main():
    disc_O = discriminator(in_channels=3, features=64).to(config.device)
    disc_A = discriminator(in_channels=3, features=64).to(config.device)
    gen_O = Generator(in_channels=3, features=64).to(config.device)
    gen_A = Generator(in_channels=3, features=64).to(config.device)
    opt_disc = optim.Adam(
        list(disc_O.parameters())+list(disc_A.parameters()),
        lr = config.learning_rate,
        betas=(0.5,0.999),
    )
    opt_gen = optim.Adam(
        list(gen_O.parameters()) + list(gen_A.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_I,gen_O, opt_gen, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_GEN_D, gen_A, opt_gen, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_CRITIC_I, disc_O, opt_disc, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_CRITIC_D, disc_A, opt_disc, config.learning_rate,)

    dataset = Color_enhancement(orange_root=config.train_dir + '/orange', apple_root=config.train_dir + '/apple', transform=config.transform)
    val_dataset = Color_enhancement(orange_root=config.val_dir + '/orange', apple_root=config.val_dir + '/apple', transform=config.transform)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,drop_last=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.num_epochs):
        train(disc_O, disc_A, gen_A, gen_O, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.save_model:
            save_checkpoint(gen_O, opt_gen, filename=config.CHECKPOINT_GEN_I)
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(disc_O, opt_disc, filename=config.CHECKPOINT_CRITIC_I)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_D)

if __name__ == '__main__':
    main()