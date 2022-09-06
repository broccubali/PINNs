"""Run a PINN example"""

import torch
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import sys
import numpy as np

from pinn import PhysicsInformedNN, nn_grad

# Create data for training
NUMPS = 50
x1    = np.linspace(-1, 1, num=NUMPS, dtype=np.float32)
x2    = np.linspace(-1, 1, num=NUMPS, dtype=np.float32)
zs    = np.zeros(NUMPS, dtype=np.float32).reshape(-1, 1)
xt1, xt2 = np.meshgrid(x1, x2, indexing='ij')
xt1 = xt1.flatten().reshape(-1, 1)
xt2 = xt2.flatten().reshape(-1, 1)

yt1 = np.sin(np.pi*xt1) + np.cos(np.pi*xt2)
yt2 = xt1**3 + xt2**2

X  = np.concatenate((xt1, xt2), 1)
Y  = np.concatenate((yt1, yt2), 1)
xt = torch.from_numpy(X).float()
yt = torch.from_numpy(Y).float()

training_data = data_utils.TensorDataset(xt, yt)
train_loader  = data_utils.DataLoader(training_data,
                                      batch_size=32,
                                      shuffle=True)


# Define pde and PINN
class TestPINN(PhysicsInformedNN):
    '''Test PINN'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pde(self, out, coords):
        dfdt = nn_grad(out[:, 0], coords)
        dgdt = nn_grad(out[:, 1], coords)
        invf = self.inv_fields[0](coords)

        eq1 = dfdt[:, 0] - np.pi*torch.cos(np.pi*coords[:, 0])
        eq2 = invf[:, 0] + np.pi*torch.sin(np.pi*coords[:, 1])
        eq3 = dgdt[:, 0] - self.inv_ctes[1]*coords[:, 0]**2
        eq4 = dgdt[:, 1] - self.inv_ctes[0]*coords[:, 1]

        return [eq1, eq2, eq3, eq4]


def main(which):
    """main train function"""

    # Set network
    if which == 'elu':
        main_kwargs = {'activation': 'elu'}
        lr          = 1e-3
    elif which == 'siren':
        main_kwargs = {'activation': 'siren', 'first_omega_0': 5.0, 'hidden_omega_0': 5.0}
        lr          = 1e-4

    # Instantiate model
    inv_kwargs  = {'mask': [0, 1]}
    PINN = TestPINN([2, 2, 1, 64],
                    inv_ctes=[1.0, 1.0],
                    lphys={'value': 1.0, 'rule': 'adam-like'},
                    inv_fields=[([2, 1, 1, 32], inv_kwargs)],
                    nn_kwargs=main_kwargs,
                    lr=lr)

    # Create Trainer with checkpointing and logging
    checkpoint_callback = ModelCheckpoint(dirpath='ckpt',
                                          save_last=True,
                                          save_top_k=5,
                                          monitor='epoch',
                                          mode='max',
                                          )
    logger = CSVLogger(save_dir='logs', name='')
    trainer = pl.Trainer(max_epochs=100,
                         enable_progress_bar=False,
                         callbacks=[checkpoint_callback],
                         logger=[logger],
                         )

    # Train
    trainer.fit(model=PINN, train_dataloaders=train_loader)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        which = sys.argv[1]
    else:
        which = 'elu'
    print(f'Running test for {which}')
    main(which)