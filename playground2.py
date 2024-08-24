from pathlib import Path
import pickle as pkl

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

mdl_test_ds = pkl.loads(Path('model_test_data.pkl').read_bytes())

model = mdl_test_ds['model']
test_ds = mdl_test_ds['ds']
test_data = mdl_test_ds['test_data']


def filter_test_data(test_data, patients_project_id):
    filtered_data = {}
    for key, data in test_data.items():
        # Find indices of the patients in patients_project_id
        indices = [i for i, patient in enumerate(data['patient']) if patient in patients_project_id]

        # Filter each component of the data using the selected indices
        filtered_data[key] = {k: [v[i] for i in indices] for k, v in data.items()}

        for key_ in ['features', 'surv_fn', 'duration']:
            try:
                filtered_data[key][key_] = torch.stack(filtered_data[key][key_])
            except RuntimeError:
                pass
    modalities = list(filtered_data.keys())
    for key in modalities:
        if not len(filtered_data[key]['patient']):
            del filtered_data[key]
    return filtered_data


project_id_concordance = {}
for project_id in tqdm(sorted(list(test_ds.datasets['Clinical'].data['project_id'].unique()))):
    patients_project_id = test_ds.datasets['Clinical'].data[test_ds.datasets['Clinical'].data['project_id'] == project_id].index
    # model.evaluate_concordance_loss(test_data)
    filtered_test_data = filter_test_data(test_data, patients_project_id)
    try:
        project_id_concordance[project_id] = model.evaluate_concordance_loss(filtered_test_data)
    except ZeroDivisionError:
        pass

pass

