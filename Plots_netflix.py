from torch import tensor
import torch
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame

baseline = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0793, 0.0768, 0.0748, 0.0729,
        0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0202, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
        0.0393], device='cuda:0')}
CPMF = {'RMSE': tensor(0.8647, device='cuda:0'), 'Precision': tensor([0.0832, 0.0763, 0.0718, 0.0683, 0.0655, 0.0632, 0.0612, 0.0596, 0.0582,
        0.0569], device='cuda:0'), 'Recall': tensor([0.0055, 0.0098, 0.0136, 0.0169, 0.0200, 0.0229, 0.0256, 0.0282, 0.0307,
        0.0330], device='cuda:0'), 'Quantile RMSE': tensor([0.6807, 0.7074, 0.7196, 0.7315, 0.7433, 0.7573, 0.7712, 0.7856, 0.8005,
        0.8169, 0.8327, 0.8518, 0.8720, 0.8926, 0.9166, 0.9452, 0.9786, 1.0219,
        1.0837, 1.1750]), 'Quantile MAP': tensor([0.0423, 0.0542, 0.0606, 0.0635, 0.0662, 0.0682, 0.0686, 0.0687, 0.0680,
        0.0693, 0.0667, 0.0666, 0.0637, 0.0625, 0.0578, 0.0548, 0.0482, 0.0419,
        0.0299, 0.0159]), 'RRI': tensor([0.0943, 0.1599, 0.2042, 0.2370, 0.2600, 0.2790, 0.2925, 0.3022, 0.3121]),
        'Correlation': (tensor(0.1464, device='cuda:0'), tensor(0.1820, device='cuda:0')), 'RPI': tensor(0.7914, device='cuda:0'), 'Classification': (-0, 0.6165257389290548)}
OrdRec = {'RMSE': tensor(1.0191, device='cuda:0', dtype=torch.float64), 'Precision': tensor([0.0635, 0.0602, 0.0564, 0.0578, 0.0573, 0.0550, 0.0528, 0.0508, 0.0487,
        0.0468], device='cuda:0'), 'Recall': tensor([0.0027, 0.0048, 0.0066, 0.0096, 0.0123, 0.0143, 0.0162, 0.0181, 0.0197,
        0.0211], device='cuda:0'), 'Quantile RMSE': tensor([0.9893, 0.9950, 0.9690, 0.9629, 0.9660, 0.9733, 0.9831, 0.9932, 1.0023,
        1.0074, 1.0069, 1.0006, 0.9835, 0.9432, 1.0177, 1.0775, 1.0905, 1.0954,
        1.1152, 1.1775]), 'Quantile MAP': tensor([0.0551, 0.0557, 0.0239, 0.0181, 0.0542, 0.0617, 0.0602, 0.0587, 0.0561,
        0.0522, 0.0521, 0.0490, 0.0467, 0.0451, 0.0448, 0.0454, 0.0499, 0.0599,
        0.0356, 0.0107]), 'RRI': tensor([ 0.0407,  0.1000, -0.0223,  0.0032,  0.0686,  0.0990,  0.1231,  0.1568,
         0.1883]), 'Correlation': (tensor(0.1400, device='cuda:0', dtype=torch.float64), tensor(0.3095, device='cuda:0', dtype=torch.float64)), 'RPI': tensor(0.1661, device='cuda:0', dtype=torch.float64),
          'Classification': (-0.53819117825272, 0.5794726776364477)}
Ensemble = {'RMSE': tensor(0.8319, device='cuda:0'), 'Precision': tensor([0.1214, 0.1113, 0.1033, 0.0973, 0.0925, 0.0885, 0.0852, 0.0824, 0.0799,
        0.0776], device='cuda:0'), 'Recall': tensor([0.0077, 0.0136, 0.0183, 0.0225, 0.0264, 0.0299, 0.0331, 0.0362, 0.0391,
        0.0419], device='cuda:0'), 'Quantile RMSE': tensor([0.7707, 0.7737, 0.7772, 0.7817, 0.7858, 0.7904, 0.7949, 0.8015, 0.8055,
        0.8116, 0.8183, 0.8248, 0.8309, 0.8382, 0.8471, 0.8580, 0.8708, 0.8887,
        0.9193, 1.0058]), 'Quantile MAP': tensor([0.0094, 0.0144, 0.0200, 0.0269, 0.0344, 0.0437, 0.0508, 0.0599, 0.0674,
        0.0767, 0.0843, 0.0925, 0.0975, 0.1087, 0.1159, 0.1204, 0.1245, 0.1342,
        0.1380, 0.1336]), 'RRI': tensor([-0.0661, -0.1171, -0.1595, -0.1898, -0.2149, -0.2349, -0.2530, -0.2687,
        -0.2826]), 'Correlation': (tensor(0.0911, device='cuda:0'), tensor(0.0704, device='cuda:0')), 'RPI': tensor(0.3794, device='cuda:0'), 'Classification': (-0.5088254632126619, 0.5465176065822879)}
Resample = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
        0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
        0.0393], device='cuda:0'), 'Quantile RMSE': tensor([0.7638, 0.7785, 0.7888, 0.7966, 0.8053, 0.8130, 0.8200, 0.8263, 0.8322,
        0.8389, 0.8462, 0.8536, 0.8599, 0.8690, 0.8791, 0.8929, 0.9148, 0.9444,
        0.9885, 1.1038]), 'Quantile MAP': tensor([0.0151, 0.0258, 0.0336, 0.0415, 0.0475, 0.0519, 0.0576, 0.0622, 0.0663,
        0.0712, 0.0753, 0.0799, 0.0825, 0.0877, 0.0939, 0.0984, 0.1024, 0.1080,
        0.1145, 0.1095]), 'RRI': tensor([-0.0663, -0.1230, -0.1675, -0.1999, -0.2267, -0.2472, -0.2637, -0.2801,
        -0.2954]), 'Correlation': (tensor(0.1168, device='cuda:0'), tensor(0.0883, device='cuda:0')), 'RPI': tensor(0.5148, device='cuda:0'), 'Classification': (-0.526729740709993, 0.558428172413906)}
usup = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
        0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
        0.0393], device='cuda:0'), 'Quantile RMSE': tensor([0.8541, 0.8662, 0.8749, 0.8536, 0.8682, 0.8677, 0.8582, 0.8557, 0.8521,
        0.8457, 0.8493, 0.8706, 0.8437, 0.8952, 0.8843, 0.8523, 0.8575, 0.9003,
        0.8659, 0.8743]), 'Quantile MAP': tensor([0.0801, 0.0797, 0.0761, 0.0782, 0.0786, 0.0803, 0.0769, 0.0764, 0.0741,
        0.0705, 0.0740, 0.0731, 0.0721, 0.0684, 0.0665, 0.0666, 0.0621, 0.0577,
        0.0576, 0.0558]), 'RRI': tensor([0.0029, 0.0084, 0.0132, 0.0140, 0.0148, 0.0157, 0.0153, 0.0159, 0.0151]), 'Correlation': (tensor(0.0037, device='cuda:0'), tensor(0.0053, device='cuda:0')), 'RPI': tensor(0.0180, device='cuda:0'), 'Classification': (-0.5309353203483685, 0.503729181203347)}
isup = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
        0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
        0.0393], device='cuda:0'), 'Quantile RMSE': tensor([0.9316, 0.8911, 0.9186, 0.9011, 0.8862, 0.8684, 0.8463, 0.8704, 0.8523,
        0.8447, 0.8268, 0.8320, 0.8308, 0.8204, 0.8287, 0.8341, 0.8423, 0.8528,
        0.8763, 0.9246]), 'Quantile MAP': tensor([0.0955, 0.0952, 0.0948, 0.0964, 0.0951, 0.0946, 0.0919, 0.0894, 0.0867,
        0.0807, 0.0795, 0.0744, 0.0679, 0.0640, 0.0569, 0.0477, 0.0424, 0.0346,
        0.0258, 0.0112]), 'RRI': tensor([0.1768, 0.2907, 0.3752, 0.4391, 0.4910, 0.5317, 0.5677, 0.5975, 0.6240]), 'Correlation': (tensor(-0.0261, device='cuda:0'), tensor(-0.0113, device='cuda:0')), 'RPI': tensor(-0.1172, device='cuda:0'), 'Classification': (-0.6033745291862661, 0.4917785719107094)}
ivar = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
        0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
        0.0393], device='cuda:0'), 'Quantile RMSE': tensor([1.0506, 0.9697, 0.9602, 0.9341, 0.9132, 0.9092, 0.8877, 0.8784, 0.8558,
        0.8630, 0.8572, 0.8451, 0.8302, 0.8256, 0.8116, 0.7921, 0.7968, 0.7704,
        0.7515, 0.7173]), 'Quantile MAP': tensor([0.0712, 0.0854, 0.0835, 0.0823, 0.0820, 0.0819, 0.0836, 0.0849, 0.0808,
        0.0812, 0.0797, 0.0760, 0.0731, 0.0695, 0.0661, 0.0601, 0.0549, 0.0495,
        0.0433, 0.0358]), 'RRI': tensor([-0.0105,  0.0021,  0.0120,  0.0152,  0.0191,  0.0240,  0.0241,  0.0226,
         0.0232]), 'Correlation': (tensor(-0.1147, device='cuda:0'), tensor(-0.0956, device='cuda:0')), 'RPI': tensor(-0.4536, device='cuda:0'), 'Classification': (-0.5262232536814566, 0.566632401794646)}
funksvdcv = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
             0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
             0.0393], device='cuda:0'), 'Quantile RMSE': tensor([0.8430, 0.7080, 0.7042, 0.7153, 0.7287, 0.7441, 0.7606, 0.7757, 0.7916,
             0.8083, 0.8265, 0.8443, 0.8649, 0.8862, 0.9115, 0.9389, 0.9702, 1.0146,
             1.0767, 1.2076]), 'Quantile MAP': tensor([0.0081, 0.0166, 0.0282, 0.0436, 0.0607, 0.0700, 0.0801, 0.0853, 0.0887,
             0.0913, 0.0930, 0.0944, 0.0939, 0.0923, 0.0916, 0.0894, 0.0856, 0.0808,
             0.0737, 0.0572]), 'RRI': tensor([-0.0152, -0.0380, -0.0555, -0.0713, -0.0858, -0.0990, -0.1100, -0.1189,
             -0.1273]), 'Correlation': (tensor(0.1643, device='cuda:0'), tensor(0.1591, device='cuda:0')), 'RPI': tensor(0.6628, device='cuda:0'), 'Classification': (-0.5214437415234605, 0.6040501355273163)}
biascv = {'RMSE': tensor(0.8646, device='cuda:0'), 'Precision': tensor([0.1041, 0.0961, 0.0903, 0.0858, 0.0823, 0.0792, 0.0768, 0.0748, 0.0729,
          0.0712], device='cuda:0'), 'Recall': tensor([0.0065, 0.0117, 0.0162, 0.0201, 0.0238, 0.0272, 0.0304, 0.0335, 0.0365,
          0.0393], device='cuda:0'), 'Quantile RMSE': tensor([0.5740, 0.6304, 0.6619, 0.6880, 0.7130, 0.7327, 0.7549, 0.7763, 0.7961,
          0.8169, 0.8374, 0.8596, 0.8824, 0.9066, 0.9357, 0.9673, 1.0063, 1.0523,
          1.1201, 1.2561]), 'Quantile MAP': tensor([0.0641, 0.0790, 0.0827, 0.0838, 0.0835, 0.0823, 0.0820, 0.0813, 0.0794,
          0.0790, 0.0760, 0.0743, 0.0724, 0.0689, 0.0674, 0.0628, 0.0594, 0.0550,
          0.0501, 0.0413]), 'RRI': tensor([ 0.0018, -0.0206, -0.0397, -0.0498, -0.0594, -0.0684, -0.0752, -0.0783,
          -0.0816]), 'Correlation': (tensor(0.2534, device='cuda:0'), tensor(0.2221, device='cuda:0')), 'RPI': tensor(0.9728, device='cuda:0'), 'Classification': (-0.508943102704638, 0.6466635989126877)}

evaluation = {'Baseline': baseline,
              'User support': usup,
              'Item support': isup,
              'Item variance': ivar,
              'FunkSVD-CV': funksvdcv,
              'Bias-CV': biascv,
              'CPMF': CPMF,
              'OrdRec': OrdRec,
              'Ensemble': Ensemble,
              'Resample': Resample}

keys = ['Baseline', 'Ensemble', 'CPMF', 'OrdRec']
rmse = [evaluation[key]['RMSE'].item() for key in keys]
DataFrame(rmse, index=keys, columns=['RMSE'])

color=iter(plt.cm.rainbow(np.linspace(0, 1, 4)))
f, ax = plt.subplots(nrows=2, figsize=(5, 10))
for key in ['Baseline', 'Ensemble', 'CPMF', 'OrdRec']:
    c = next(color)
    ax[0].plot(np.arange(1, 11), evaluation[key]['Precision'].cpu().detach().numpy(), '-', color=c, label=key)
    ax[1].plot(np.arange(1, 11), evaluation[key]['Recall'].cpu().detach().numpy(), '-', color=c, label=key)
ax[0].set_xticks(np.arange(1, 11))
ax[0].set_xlabel('K', Fontsize=20)
ax[0].set_ylabel('Average precision at K', Fontsize=20)
ax[0].legend(ncol=2)
ax[1].set_xticks(np.arange(1, 11))
ax[1].set_xticklabels(np.arange(1, 11))
ax[1].set_xlabel('K', Fontsize=20)
ax[1].set_ylabel('Average recall at K', Fontsize=20)
ax[1].legend(ncol=2)
f.tight_layout()
f.savefig('Empirical study/Netflix/precision_recall.pdf')

keys = list(evaluation.keys())[1:]
out = DataFrame(np.zeros((3, 9)), index=['RPI', 'Pearson', 'Spearman'], columns=keys)
for key in keys:
    out[key] = (evaluation[key]['RPI'].item(),
                evaluation[key]['Correlation'][0].item(),
                evaluation[key]['Correlation'][1].item())
out.T

















