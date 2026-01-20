import torch
import torchvision.transforms.functional as TF
import random

def plot_vae_samples(vae, visdom_session):
    vis = visdom_session
    x = torch.zeros([1, 784])
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(1, 28, 28).detach().cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)

def mnist_test_tsne(vae=None, test_loader=None, equivariant=False):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = "VAE"
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, mnist_labels, name, equivariant=equivariant)


def plot_tsne(z_loc, classes, name, equivariant=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        
        #new fix:
        #if the classes are one-hot encoded, they become two dimensional
        #where dimension 0 is the sample index and dimension 1 is the class index (0-9)
        #argmax takes care of the one-hot encoding by taking the index of the single 1, which is the class label for MNIST
        
        # if using a different dataset with different label structure, this logic will need to be changed.
        if classes.ndim == 2:
            classes = classes.argmax(axis=1)
        ind_class = (classes == ic)
        #old but this breaks:
            # ind_vec[:, ic] = 1
            # ind_class = classes[:, ic] == 1

        #new but this breaks too:
        # try:
        #     ind_vec[:, ic] = 1
        # except IndexError:
        #     ind_vec[ic] = 1
        # try:
        #     ind_class = classes[:, ic] == 1
        # except IndexError:
        #     ind_class = classes[ic] == 1

        # previously code did not have enough distinct colors for all classes:
        # plt.cm.Set1(ic)
        color = plt.cm.tab10(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color, label=f'Digit {ic}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Latent Variable T-SNE per Class {'(Equivariant Model)' if equivariant else ''}")
        fig.savefig("./vae_results/" + str(name) + "_embedding_" + f"{'equivariant_' if equivariant else ''}" + str(ic) + ".png", bbox_inches='tight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig("./vae_results/" + str(name) + "_embedding" + f"{'_equivariant' if equivariant else ''}" + ".png", bbox_inches='tight')

def plot_llk(train_elbo, test_elbo, equivariant=False):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    Path("vae_results").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    df1 = pd.DataFrame(
        {
            "Epoch": train_elbo.keys(),
            "Loss": [-val for val in train_elbo.values()],
            "dataset": "Train",
        }
    )
    df2 = pd.DataFrame(
        {
            "Epoch": test_elbo.keys(),
            "Loss": [-val for val in test_elbo.values()],
            "dataset": "Test",
        }
    )
    df = pd.concat([df1, df2], axis=0)

    # Create the FacetGrid with scatter plot
    g = sns.FacetGrid(df, height=4, aspect=1.5, hue="dataset")
    g.map(sns.scatterplot, "Epoch", "Loss")
    g.map(sns.lineplot, "Epoch", "Loss", linestyle="--")
    g.ax.yaxis.get_major_locator().set_params(integer=True)
    g.add_legend()
    plt.savefig("./vae_results/test_elbo_vae" + f"{'_equivariant' if equivariant else ''}" + ".png")
    plt.close("all")

def plot_equivariance_loss(equivariant_loss, latent_equivariant_loss):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    Path("vae_results").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    df1 = pd.DataFrame(
        {
            "Epoch": equivariant_loss.keys(),
            "Loss": list(equivariant_loss.values()),
            "loss_type": "Equivariant Loss",
        }
    )
    df2 = pd.DataFrame(
        {
            "Epoch": latent_equivariant_loss.keys(),
            "Loss": list(latent_equivariant_loss.values()),
            "loss_type": "Latent Equivariant Loss",
        }
    )
    df = pd.concat([df1, df2], axis=0)

    # Create the FacetGrid with scatter plot
    g = sns.FacetGrid(df, height=4, aspect=1.5, hue="loss_type")
    g.map(sns.scatterplot, "Epoch", "Loss")
    g.map(sns.lineplot, "Epoch", "Loss", linestyle="--")
    g.ax.yaxis.get_major_locator().set_params(integer=True)
    g.add_legend()
    plt.savefig("./vae_results/equivariance_loss.png")
    plt.close("all")

class RandomRotation:
    def __init__(self, angles=None):
        self.angles = angles if angles is not None else [0, 180]

    def __call__(self, x):
        random.seed(0)
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)