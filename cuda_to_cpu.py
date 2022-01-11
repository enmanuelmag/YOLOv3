import pickle as pkl
import pandas as pd

loos = pkl.load(open('./loss/20220110_160349_all_looses.pkl', 'rb'))
metrics = pkl.load(open('./loss/20220110_160349_all_metrics.pkl', 'rb'))

#convert array of dict to dataframe
loos = pd.DataFrame(loos)
metrics = pd.DataFrame(metrics)
for column in metrics.columns:
  metrics[column] = metrics[column].astype(float)

print(loos.head())
print(metrics.head())

loos.to_pickle('./data/losses.pkl')
metrics.to_pickle('./data/metrics.pkl')

loos.to_csv('./data/losses.csv')
metrics.to_csv('./data/metrics.csv')