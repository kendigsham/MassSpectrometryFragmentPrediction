import pandas as pd

def read_csv(filename):
  csv_file = pd.read_csv(filename)
  return csv_file

def auc_ranked_list(csv_file, num_of_molecule, num_of_positive, auc_score):
  sorted_csv = csv_file.sort_values(by = 'auc_score', ascending = 0)
  filter_mol =  sorted_csv[(sorted_csv['num_of_molecule']>num_of_molecule) ]
  filter_mol = filter_mol[(filter_mol['num_of_positive']>num_of_positive)]
  filter_mol = filter_mol[(filter_mol['auc_score']>auc_score)]
#  print filter_mol
  filter_mol.to_csv('gnps_20_10_0.8.csv',index=False)
  
  mz_values= filter_mol.iloc[:]['m/z_value'].tolist()
  mz_value_list = map(lambda x: "{:.3f}".format(x), mz_values)
  return mz_value_list

def get_filtered_dataframe(filename, num_of_molecule, num_of_positive, auc_score):
  csv_file = pd.read_csv(filename)
  sorted_csv = csv_file.sort_values(by = 'auc_score', ascending = 0)
  filter_mol =  sorted_csv[(sorted_csv['num_of_molecule']>num_of_molecule) ]
  filter_mol = filter_mol[(filter_mol['num_of_positive']>num_of_positive)]
  filter_mol = filter_mol[(filter_mol['auc_score']>auc_score)]
  return filter_mol

def filtered_csv(csv_file, num_of_molecule, num_of_positive, auc_score):

  filter_mol =  csv_file[(csv_file['num_of_molecule']>num_of_molecule) ]

  filter_mol = filter_mol[(filter_mol['num_of_positive']>num_of_positive)]
  
  filter_mol = filter_mol[(filter_mol['auc_score']>auc_score)]

  return filter_mol


def get_mz_list(filename, num_of_molecule, num_of_positive, auc_score):
  csv_file = read_csv(filename)
#  mz_value_list = auc_ranked_list(csv_file, num_of_molecule, num_of_positive, auc_score)
  auc_ranked_list(csv_file, num_of_molecule, num_of_positive, auc_score)
#  return mz_value_list

get_mz_list('gnpsresult2.csv',20,10,0.8)


