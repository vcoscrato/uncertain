import torch
import numpy as np
from pandas import factorize

from uncertain.models.explicit.ExplicitFactorizationModel import ExplicitFactorizationModel
from uncertain.models.explicit.CPMF import CPMF
from uncertain.models.explicit.OrdRec import OrdRec
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split

from utils import evaluate

data = get_movielens_dataset(variant='10M')
train, test = random_train_test_split(data, test_percentage=0.2)
train = train.gpu()
test = test.gpu()


fold1, fold2 = random_train_test_split(train, random_state=0, test_percentage=0.5)
model_cv = deepcopy(models['Baseline'])
model_cv._initialize(fold1)
model_cv.fit(fold1)
predictions1 = model_cv.predict(fold2.user_ids, fold2.item_ids)
model_cv._initialize(fold2)
model_cv.fit(fold2)
predictions2 = model_cv.predict(fold1.user_ids, fold1.item_ids)
train_errors = torch.cat((torch.abs(fold2.ratings - predictions1), torch.abs(fold1.ratings - predictions2)))
train_errors = Interactions(torch.cat((fold2.user_ids, fold1.user_ids)),
                            torch.cat((fold2.item_ids, fold1.item_ids)),
                            train_errors, num_users=train.num_users, num_items=train.num_items)