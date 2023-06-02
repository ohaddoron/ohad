# %%
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# %%
df: pd.DataFrame = pd.read_csv(Path('/mnt/drive1/home/ohaddoron1/Projects/ohad/surv_pred/Clinical.csv').resolve())

df.drop('patient', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
X = df.to_dict('r')
y = list(range(len(X)))
# %%
net = TabNetClassifier(input_dim=10,
                       output_dim=len(y),
                       n_d=64,
                       n_a=64,
                       cat_idxs=list(range(9)),
                       cat_dims=[item[0] for item in [(33, 17), (2, 1), (8, 4), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2),
                                                      (20, 10)]],
                       cat_emb_dim=[item[1] for item in
                                    [(33, 17), (2, 1), (8, 4), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2),
                                     (20, 10)]],
                       device_name='cuda',
                       )
# %%
X_train = pd.DataFrame(X).values
y_train = np.array(y)
# %%
net.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric=['accuracy', 'logloss'], max_epochs=10000,
        patience=-1, from_unsupervised='weight_checkpoint_2023-2-4_19-15-36_BiYEt46PoRrdb7179T53.pt', warm_start=True)
