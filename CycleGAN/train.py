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

def train(disc_I, disc_D, gen_I, gen_D, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    I_reals = 0
    I_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (iphone, dsrl) in enumerate(loop):
        iphone = iphone.to(config.device)
        dsrl = dsrl.to(config.device)
        #Train discriminator for I and D
        with torch.cuda.amp.autocast():
            fake_iphone = gen_D(iphone)
            D_I_real = disc_I(iphone)
            D_I_fake = disc_I(fake_iphone.detach())
            I_reals += D_I_real.mean().items()
            I_fakes += D_I_fake.mean().items()
            D_I_real_loss = mse(D_I_real, torch.ones_like(D_I_real))
            D_I_fake_loss = mse(D_I_fake, torch.zeros_like(D_I_fake))
            D_I_loss = D_I_real_loss + D_I_fake_loss


            fake_dsrl = gen_I(dsrl)
            D_D_real = disc_D(dsrl)
            D_D_fake = disc_D(fake_dsrl.detach())
            D_D_real_loss = mse(D_D_real, torch.ones_like(D_I_real))
            D_D_fake_loss = mse(D_D_fake, torch.zeros_like(D_D_fake))
            D_D_loss = D_D_real_loss + D_D_fake_loss
            D_loss = D_I_loss + D_D_loss
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_I_fake = disc_I(iphone)
            D_D_fake = disc_D(dsrl)
            loss_G_I = mse(D_I_fake, torch.ones_like(D_I_fake))
            loss_G_D = mse(D_D_fake, torch.zeros_like(D_D_fake))

            cycle_iphone = gen_D(fake_dsrl)
            cycle_dsrl = gen_I(fake_iphone)
            cycle_iphone_loss = l1(iphone, cycle_iphone)
            cycle_dsrl_loss =l1(dsrl, cycle_dsrl)

            identity_iphone = gen_I(iphone)
            identity_dsrl = gen_D(dsrl)
            identity_iphone_loss = l1(iphone, identity_iphone)
            identity_dsrl_loss = l1(dsrl, identity_dsrl)

            G_loss = (
                loss_G_I +
                loss_G_D +
                cycle_iphone_loss * config.lambda_cycle +
                cycle_dsrl_loss * config.lambda_cycle +
                identity_iphone_loss * config.lambda_identity +
                identity_dsrl_loss * config.lambda_identity
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_iphone * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_dsrl * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=I_reals / (idx + 1), H_fake=I_fakes / (idx + 1))

def main():
    disc_I = discriminator(in_channels=3, features=64).to(config.device)
    disc_D = discriminator(in_channels=3, features=64).to(config.device)
    gen_I = Generator(in_channels=3, features=64).to(config.device)
    gen_D = Generator(in_channels=3, features=64).to(config.device)
    opt_disc = optim.Adam(
        list(disc_I.parameters())+list(disc_D.parameters()),
        lr = config.learning_rate,
        betas=(0.5,0.999),
    )
    opt_gen = optim.Adam(
        list(gen_I.parameters()) + list(gen_D.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_I,gen_I, opt_gen, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_GEN_D, gen_D, opt_gen, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_CRITIC_I, disc_I, opt_disc, config.learning_rate,)
        load_checkpoint(config.CHECKPOINT_CRITIC_D, disc_D, opt_disc, config.learning_rate,)

    dataset = Color_enhancement(iphone_root=config.train_dir+'/iphone', dsrl_root=config.train_dir+'/dsrl', transform=config.transform)
    val_dataset = Color_enhancement(iphone_root=config.val_dir + '/iphone', dsrl_root=config.val_dir + '/dsrl', transform=config.transform)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.num_epochs):
        train(disc_I, disc_D, gen_D, gen_I, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.save_model:
            save_checkpoint(gen_I, opt_gen, filename=config.CHECKPOINT_GEN_I)
            save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(disc_I, opt_disc, filename=config.CHECKPOINT_CRITIC_I)
            save_checkpoint(disc_D, opt_disc, filename=config.CHECKPOINT_CRITIC_D)

if __name__ == '__main__':
    main()