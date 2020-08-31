print('Comparing all the methods...')
eval = df({'RMSE': [ensemble.rmse[-1]] + [R.rmse] * 3, 'Error RMSE': [0] * 2 + [double.ermse, linear.ermse],
           'EPI': [ensemble.epi[-1], resample.epi[-1], double.epi, linear.epi]},
          index=['Ensemble', 'Resample', 'Double', 'Linear'])
eval.round(4).to_csv(path + 'comparison.txt', index=True, header=True)

f, ax = plt.subplots(nrows=3, figsize=(10, 10))
ax[0].plot(np.arange(1, 21), ensemble.quantiles, 'b-', label='Ensemble')
ax[0].plot(np.arange(1, 21), resample.quantiles, 'g-', label='Resample')
ax[0].plot(np.arange(1, 21), double.quantiles, 'r-', label='Double')
ax[0].plot(np.arange(1, 21), linear.quantiles, 'k-', label='Linear')
ax[0].set_xticks(np.arange(1, 21))
ax[0].set_xticklabels(np.round(np.linspace(start=0, stop=1, num=21, endpoint=True), 2))
ax[0].set_xlabel('Error bin', Fontsize=20)
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend()
ax[1].plot(np.arange(1, 21), ensemble.intervals, 'b-', label='Ensemble')
ax[1].plot(np.arange(1, 21), resample.intervals, 'g-', label='Resample')
ax[1].plot(np.arange(1, 21), double.intervals, 'r-', label='Double')
ax[1].plot(np.arange(1, 21), linear.intervals, 'k-', label='Linear')
ax[1].set_xticks(np.arange(1, 21))
ax[1].set_xticklabels(np.arange(1, 21))
ax[1].set_xlabel('Error bin', Fontsize=20)
ax[1].set_ylabel(r'$\epsilon$', Fontsize=20)
ax[1].legend()
ax[2].plot(k, ensemble.eri, 'b-', label='Ensemble')
ax[2].plot(k, resample.eri, 'g-', label='Resample')
ax[2].plot(k, double.eri, 'r-', label='Double')
ax[2].plot(k, linear.eri, 'k-', label='Linear')
ax[2].set_xticks(k)
ax[2].set_xticklabels(k)
ax[2].set_xlabel('K', Fontsize=20)
ax[2].set_ylabel('ERI@K', Fontsize=20)
ax[2].legend()
f.tight_layout()
f.savefig(path + 'comparison.pdf')

'''
eval['Heuristic'] = [R.RMSE, np.nan, models['Heuristic'].RPI]
eval['EMF'] = [models['EMF'].val_metrics['RMSE'][-1], np.nan, models['EMF'].val_metrics['RPI'][-1]]
eval['RMF'] = [models['RMF'].val_metrics['RMSE'], np.nan, models['RMF'].val_metrics['RPI'][-1]]
eval['Linear'] = [models['Linear'].RMSE, models['Linear'].ERMSE, models['Linear'].RPI]
eval['NMF'] = [models['NMF'].RMSE, models['NMF'].ERMSE, models['NMF'].RPI]
eval = df(eval, index=['RMSE', 'ERMSE', 'RPI'])
print('Full test data metrics:\n', eval.T, '\n')
predictions_ = {}
predictions_['Heuristic'] = [a for a in predictions['Heuristic'] if 4 < a.est < 5]
predictions_['EMF'] = [a for a in predictions['EMF'] if 4 < a.est < 5]
predictions_['RMF'] = [a for a in predictions['RMF'] if 4 < a.est < 5]
predictions_['Linear'] = [a for a in predictions['Linear'] if 4 < a.est < 5]
predictions_['NMF'] = [a for a in predictions['NMF'] if 4 < a.est < 5]
test_error_ = [test_error[i] for i in range(len(test_error)) if 4 < predictions['Linear'][i].est < 5]
eval['Heuristic'] = [rmse(predictions_['Heuristic']), np.nan, RPI(predictions_['Heuristic'])]
eval['EMF'] = [rmse(predictions_['EMF']), np.nan, RPI(predictions_['EMF'])]
eval['RMF'] = [rmse(predictions_['RMF']), np.nan, RPI(predictions_['RMF'])]
eval['Linear'] = [rmse(predictions_['Linear']), srmse(models['Linear'].E.test(test_error_), verbose=False), RPI(predictions_['Linear'])]
eval['NMF'] = [rmse(predictions_['NMF']), srmse(models['NMF'].E.test(test_error_), verbose=False), RPI(predictions_['NMF'])]
eval = df(eval, index=['RMSE', 'ERMSE', 'RPI'])
print('Metrics for ratings in [4, 5]: \n {}'.format(eval.T, '\n'))
'''

f, ax = plt.subplots(figsize=(10, 5))
ax.plot(k, ensemble.eri, 'b-', label='Ensemble')
ax.plot(k, resample.eri, 'g-', label='Resample')
ax.plot(k, double.eri, 'r-', label='Double')
ax.plot(k, linear.eri, 'k-', label='Linear')
ax.set_xticks(k)
ax.set_xticklabels(k)
ax.set_xlabel('K', Fontsize=20)
ax.set_ylabel('URI@K', Fontsize=20)
ax.legend()
f.tight_layout()

f, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(range(1, n_models + 1), ensemble.rmse, 'b-', label='Ensemble')
ax[0].plot(range(1, n_models + 1), [ensemble.rmse[0]] * n_models, 'b--', label='Baseline')
ax[0].set_xlabel('Number of models', Fontsize=20, labelpad=10)
ax[0].set_xticks(range(1, n_models + 1))
ax[0].set_xticklabels(range(1, n_models + 1))
ax[0].set_ylabel('RMSE', Fontsize=20)
ax[0].legend()
ax[1].plot(k, ensemble.precision, 'r-', label='Ensemble precision')
ax[1].plot(k, R.precision, 'r--', label='Baseline precision')
ax[1].plot(k, ensemble.recall, 'g-', label='Ensemble recall')
ax[1].plot(k, R.recall, 'g--', label='Baseline recall')
ax[1].set_xticks(k)
ax[1].set_xticklabels(k)
ax[1].set_xlabel('K', Fontsize=20, labelpad=10)
ax[1].set_ylabel('Metric@K', Fontsize=20)
ax[1].legend()
f.tight_layout()
f.savefig(path + 'ensemble vs baseline.pdf')