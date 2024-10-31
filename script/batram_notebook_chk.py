# Read example data from NetCDF
from netCDF4 import Dataset

# Packages for working with array data and tensors
import numpy as np
import matplotlib.pyplot as plt
import torch

# Packages for building transport maps
import veccs.orderings
from batram.legmods import Data, SimpleTM

# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

# TODO: Get a prettier example to use as the feature in the eventual repo code.
# Either reproduce data from the main paper or get a nice precip dataset

# Load data and print dimensions
dtfl = '../config/batram_examp_fields.nc'
ncbr = Dataset(dtfl,'r')
locs = ncbr.variables['loc_coord'][:,:]
obsdt = ncbr.variables['sim_data'][:,:]
ncbr.close()

obs = torch.as_tensor(obsdt)

print(f"Locations array dimension: {locs.shape}")
print(f"Observations array dimension: {obs.shape}")

# The loaded data contained 200 replicate spatial fields (200 samples, 900 locs).
# We will subset the data to use the first 10 samples and the entire spatial
# field.
obs = obs[:10, :]

# Pixel-wise data centering. NB `obs` is a torch tensor.
obs = obs - obs.mean(dim=0, keepdim=True)

# Maximin ordering of the locations using the `veccs` package. 
# Note, the locations array is reordered over its first dimension, whereas the
# observations are reordered over the last dimension. 
order = veccs.orderings.maxmin_cpp(locs)
locs = locs[order, ...]
obs = obs[..., order]

# Finding nearest neighbors using the `veccs` package.
# The computation time of the model scales as a function of the condition set
# size. We recommend restricting this to be no larger than 30 neighbors.
largest_conditioning_set = 30
nn = veccs.orderings.find_nns_l2(locs, largest_conditioning_set)

# Create a `Data` object for use with the `SimpleTM` model.
# All objects passed to this class must be torch tensors, so we type convert
# the numpy arrays in this step.
data = Data.new(
    torch.as_tensor(locs),
    obs,
    torch.as_tensor(nn)
)

# TODO: Move this initialization into the constructor.
#   - We should allow the user to pass a custom initialization if they wish to
#   - We should provide by default

# Construct an initial parameter vector for the model.
#
#Note that theta[2] (theta_q in the paper) is wrapped by a response function to
# ensure the weights are decaying. Thus, we initialize it 0.0 instead of -1.0.
theta_init = torch.tensor(
    [data.response[:, 0].square().mean().log(), 0.2, 0.0, 0.0, 0.0, -1.0]
)

# Optional arguments at construction time (shown here with their default values):
# - The `smooth` parameter is the smoothness parameter (nu) of a Matern kernel.
#   We use the `gpytorch` implementation of Matern kernels, so this is required
#   to be one of 0.5, 1.5, or 2.5.
# - The `nugMult` parameter is the multiplier for the nugget term in the kernel.
#   Section 3.3 of [1] explains the default value passed here. Generally it does
#   not need to be changed.
#   [1] https://doi.org/10.1080/01621459.2023.2197158
tm = SimpleTM(data, theta_init.clone(), False, smooth=1.5, nugMult=4.0)

# Any fit requires passing the number of steps to run the optimizer for with an
# initial learning rate. The remaining arguments are optional and ignored here.
nsteps = 200
initial_learning_rate = 0.1
res = tm.fit(nsteps, initial_learning_rate)

# change optimizer
tm = SimpleTM(data, theta_init.clone(), False, smooth=1.5, nugMult=4.0)

nsteps = 200
opt = torch.optim.Adam(tm.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps)
res = tm.fit(nsteps, 0.1, test_data=tm.data, optimizer=opt, scheduler=sched, batch_size=128)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
res.plot_loss(axs[0], use_inset=False)
res.plot_loss(axs[1], use_inset=True)
plt.show()

