import os
import argparse
import time

import numpy as np
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" #this has to come before importing torch
import torch
from pyro.contrib.examples.util import MNIST
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine

import visdom
from pprint import pprint

from vae_utils import plot_vae_samples, mnist_test_tsne, plot_llk, plot_equivariance_loss, RandomRotation

assert pyro.__version__.startswith('1.9.1')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0) #sets the random seed for pyro and torch

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False, equivariant=False):
    root = './data'
    download = True
    train_trans = transforms.ToTensor()
    test_trans = transforms.ToTensor()

    # we augment the dataset with rotations for the non-equivariant model, to provide a fair comparison:
    if not equivariant:
        train_trans = transforms.Compose([RandomRotation(), transforms.ToTensor()])
    
    train_set = MNIST(root=root, train=True, transform=train_trans,
                      download=download)
    test_set = MNIST(root=root, train=False, transform=test_trans)

    kwargs = {'pin_memory': use_cuda} #removed 'num_workers': 1
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img
    

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale
    

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False, equivariant=False, lambda_eq=0, lambda_latent_eq=0):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.equivariant = equivariant
        self.lambda_eq = lambda_eq
        self.lambda_latent_eq = lambda_latent_eq

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        if self.equivariant:
            #TODO do we actually need this?
            # register PyTorch module `encoder` with Pyro, as we need it for equivariance loss
            pyro.module("encoder", self.encoder)
        #TODO why is all of this in the model and not the guide?
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))


            if self.equivariant: #and torch.rand(1).item() < 0.5:
                base = torch.tensor([[1., 0.], [0., -1.]], device=z.device, dtype=z.dtype)
                representation = torch.kron(torch.eye(self.z_dim // 2, device=z.device, dtype=z.dtype), base)

                # flatten images into vectors
                x = x.view(-1, 28 * 28)
                
                #Step 1: original task (done above):
                # not sampling here, since we are looking to transform the mean of the distribution
                # z_loc = self.encoder(x)[0]
                # x_hat_loc = self.decoder(z)
                # ----------------------

                # Step 2: the following two should be the same:
                # 1) transforming image then encoding and decoding
                x_t = torch.rot90(x.view(-1, 1, 28, 28), k=2, dims=[2, 3]).view(-1, 28 * 28)
                z_x_t = self.encoder(x_t)[0]
                x_t_e_d_loc = self.decoder(z_x_t) #extract loc only, we do not sample to not introduce probabilistic noise into equivariant structure
                # 2) encoding and decoding then transforming
                x_hat_loc_t = torch.rot90(loc_img.view(-1, 1, 28, 28), k=2, dims=[2, 3]).view(-1, 28 * 28)

                equivariance_loss = torch.nn.functional.mse_loss(x_t_e_d_loc, x_hat_loc_t)
                # ----------------------


                # Step 3: the following two should be the same:
                # 1) transforming image then encoding
                # (denoted by z_x_t above in Step 2)
                # 2) encoding image then transforming in latent space
                z_transformed = torch.matmul(z, representation)

                latent_equivariance_loss = torch.nn.functional.mse_loss(z_x_t, z_transformed)
                # ----------------------   
                
                # do ELBO gradient and accumulate loss
                # TODO explain why we use deterministic and factor here
                pyro.deterministic("equivariance_loss_value", equivariance_loss)
                pyro.deterministic("latent_equivariance_loss_value", latent_equivariance_loss)
                pyro.factor("equivariance_loss", - self.lambda_eq * equivariance_loss)
                pyro.factor("latent_equivariance_loss", - self.lambda_latent_eq * latent_equivariance_loss)

            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        #TODO what if I also register the decoder here????????
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        return z_loc, z_scale

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

def get_logged_losses(vae, x):
    # Run guide to get the latent sample, then replay model with same latent
    with torch.no_grad():
        guide_tr = poutine.trace(vae.guide).get_trace(x)
        model_tr = poutine.trace(poutine.replay(vae.model, trace=guide_tr)).get_trace(x)

        eq = model_tr.nodes.get("equivariance_loss_value", {}).get("value", None)
        leq = model_tr.nodes.get("latent_equivariance_loss_value", {}).get("value", None)

        eq_val = None if eq is None else (eq.cpu())
        leq_val = None if leq is None else (leq.cpu())

    return eq_val, leq_val

def train(svi, train_loader, use_cuda=False, vae=None):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader

    equvariance_losses = []
    latent_equvariance_losses = []

    for step, (x, _) in enumerate(train_loader):
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()

        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

        eq_loss, leq_loss = get_logged_losses(vae, x)
        if eq_loss is not None and leq_loss is not None:
            equvariance_losses.append(eq_loss)
            latent_equvariance_losses.append(leq_loss)

    equvariance_loss = torch.stack(equvariance_losses).mean() if len(equvariance_losses) > 0 else None
    latent_equvariance_loss = torch.stack(latent_equvariance_losses).mean() if len(latent_equvariance_losses) > 0 else None
    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train, equvariance_loss, latent_equvariance_loss

def evaluate(svi, test_loader, use_cuda=False, vis=None, vae=None):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for i, (x, _) in enumerate(test_loader):
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)

        # pick three random test images from the first mini-batch and
        # visualize how well we're reconstructing them
        if i == 0 and vis is not None and vae is not None:
            plot_vae_samples(vae, vis)
            reco_indices = np.random.randint(0, x.shape[0], 3)
            for index in reco_indices:
                test_img = x[index, :]
                reco_img = vae.reconstruct_img(test_img)
                vis.image(
                    test_img.reshape(28, 28).detach().cpu().numpy(),
                    opts={"caption": "test image"},
                )
                vis.image(
                    reco_img.reshape(28, 28).detach().cpu().numpy(),
                    opts={"caption": "reconstructed image"},
                )
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def sync_device():
    """helper function to synchronize gpu devices, used for timing the learning process"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def main(args):
    # clear param store: key for resetting the learning process by defaulting any learned parameters
    pyro.clear_param_store()

    #start timing
    if args.gpu_timing:
        sync_device()
    start_time = time.perf_counter()

    train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=args.cuda, equivariant=args.equivariant)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda, equivariant=args.equivariant, lambda_eq=args.lambda_eq, lambda_latent_eq=args.lambda_latent_eq)
    encoder_device = next(vae.encoder.parameters()).device
    print(f"Encoder is on device: {encoder_device}")

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = Trace_ELBO()
    train_svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    if args.equivariant:
        modified_model = poutine.block(vae.model, hide=["equivariance_loss", "latent_equivariance_loss"])
        test_svi = SVI(modified_model, vae.guide, optimizer, loss=elbo)
    else:
        test_svi = train_svi

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = {}
    test_elbo = {}
    equivariance_loss_dict = {}
    latent_equivariance_loss_dict = {}
    epoch_times = []

    if args.render_model:
        pyro.render_model(vae.model, model_args=(torch.zeros(1, 784),), filename="vae_model.png", render_distributions=True, render_params=True)
        return 
    # training loop
    for epoch in range(args.num_epochs):
        if args.gpu_timing:
            sync_device()
        epoch_start_time = time.perf_counter()
        total_epoch_loss_train, equvariance_loss, latent_equvariance_loss = train(train_svi, train_loader, use_cuda=args.cuda, vae=vae)
        train_elbo[epoch] = float(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if equvariance_loss is not None:
            equivariance_loss_dict[epoch] = float(equvariance_loss)
        print("[epoch %03d]  equivariance loss: %.4f" % (epoch, equvariance_loss)) if equvariance_loss is not None else None
        if latent_equvariance_loss is not None:
            latent_equivariance_loss_dict[epoch] = float(latent_equvariance_loss)
        print("[epoch %03d]  latent equivariance loss: %.4f" % (epoch, latent_equvariance_loss)) if latent_equvariance_loss is not None else None
        
        if args.gpu_timing:
            sync_device()
        epoch_end_time = time.perf_counter()
        print(f"Epoch {epoch} took {epoch_end_time - epoch_start_time:.4f} seconds.")
        epoch_times.append(epoch_end_time - epoch_start_time)


        if epoch % args.test_frequency == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(test_svi, test_loader, use_cuda=args.cuda, vae=vae, vis=vis if args.visdom_flag else None)
            test_elbo[epoch] = float(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            plot_llk(train_elbo, test_elbo, equivariant=args.equivariant)
            if len(equivariance_loss_dict) > 0:
                plot_equivariance_loss(equivariance_loss_dict, latent_equivariance_loss_dict)
        
        if epoch == args.tsne_iter or epoch == args.num_epochs // 2:
            mnist_test_tsne(vae=vae, test_loader=test_loader, equivariant=args.equivariant)


    if args.gpu_timing:
        sync_device()
    end_time = time.perf_counter()
    print(f"Total training time: {end_time - start_time:.4f} seconds.")

    total_time = end_time - start_time
    avg_epoch_time = np.mean(epoch_times)

    logs = {
        "train_elbo": train_elbo,
        "test_elbo": test_elbo,
        "total_time": total_time,
        "avg_epoch_time": avg_epoch_time,
        "epochs": args.num_epochs,
        "equivariant": args.equivariant,
        "lambda_eq": args.lambda_eq,
        "lambda_latent_eq": args.lambda_latent_eq,
    }
    print("Logs:\n")
    pprint(logs)
    return logs


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.1")

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-tf",
        "--test_frequency",
        default=5,
        type=int,
        help="how often we evaluate the test set",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    )

    #cuda argument is not supported in this implementation
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "-visdom",
        "--visdom_flag",
        action="store_true",
        help="Whether plotting in visdom is desired",
    )
    parser.add_argument(
        "-i-tsne",
        "--tsne_iter",
        default=100,
        type=int,
        help="epoch when tsne visualization runs",
    )

    parser.add_argument(
        "--equivariant",
        action="store_true",
        default=False,
        help="Whether to use the equivariant model",
    )

    parser.add_argument(
        "--lambda_eq",
        default=0.0,
        type=float,
        help="Weight for image space equivariance loss",
    )

    parser.add_argument(
        "--lambda_latent_eq",
        default=0.0,
        type=float,
        help="Weight for latent space equivariance loss",
    )

    parser.add_argument(
        "--render_model",
        action="store_true",
        default=False,
        help="To render model and exit",
    )

    parser.add_argument(
        "--gpu_timing",
        action="store_true",
        default=False,
        help="Whether to account for GPU synchronisation while timing",
    )

    args = parser.parse_args()

    model = main(args)