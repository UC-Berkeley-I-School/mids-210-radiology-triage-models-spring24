# Dev Notes: Tabular Model

This is primarily a recordkeeping mechanism for tracking and understand specific changes to the model (in this case the tabular model) during development.

## Initial EDA

Setting up the initial tensors and values. Having obtained the preprocessed data, we massaged the data and created a pytorch TensorDataset to be used in conjuction with the DataLoader.

There should be no issues with the labels tensor, as the binary columns naturally form a good matrix for multi-label classification, but will probably need extra work to confirm it. Example below:

| atelectasis | cardiomegaly | edema | lung_opacity | pleural_effusion | pneumonia |
|------------:|-------------:|------:|-------------:|-----------------:|----------:|
| 0           | 1            | 0     | 0            | 0                | 0         |
| 0           | 0            | 0     | 0            | 0                | 0         |
| 0           | 0            | 0     | 1            | 0                | 0         |
| 0           | 0            | 0     | 1            | 0                | 1         |
| 0           | 0            | 0     | 1            | 0                | 0         |
| 1           | 0            | 0     | 1            | 0                | 0         |
| 1           | 0            | 0     | 0            | 0                | 0         |

However, the selection of tabular features may be a problem. From the preprocessed tabular, we were only able to get the following features: `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, and `dbp`. The feature `pain` wasn't cleanly processed and is a subjective value based on the patient. `acuity` refers to the priority of the case, so should not factor into the model. Considering the remaining features, it is likely these features may not result in a predictive value of pathology finding, but rather just an indication that something deviates from the norm. This measurement is already biased by the simple fact that this is a hospital, so patients entering in already may not be within the norm.

A dedicated model may prove otherwise, however probably only an early fusion model with other features would meaningfully produce significance with the tabular set. In any case, any model will have to at least match these rates (taken from training set) to be meaningful.

Training set label outputs and rates:
| atelectasis | cardiomegaly |  edema | lung_opacity | pleural_effusion | pneumonia |
|------------:|-------------:|-------:|-------------:|-----------------:|----------:|
|     1451    | 1134         | 927    | 2190         | 1380             | 735       |
| 12.87%      | 10.06%       | 08.22% | 19.43%       | 12.24%           | 06.52%    |

## Initial Model Build Process

Initial approach is to create a multi-head multi-label classification model, with each head focused on a single class. For the time being, setup as a dummy series of linear nn.

