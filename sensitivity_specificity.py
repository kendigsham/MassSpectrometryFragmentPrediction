import read_massbank
import read_gnps
import mz_value_list
from sklearn import svm
import operator
from sklearn.metrics import confusion_matrix
import file_handler
import latent_n_class
import pickle


#%%


#model = svm.SVC(kernel='rbf',probability=True,C=0.1,gamma=10,class_weight='balanced')

classtrain, mz_value_train, smilerow_train,rowindex_train, unique_smiles_train = read_massbank.get_class_vectors()
print 'classtrain', classtrain.shape
classtest, mz_value_test, smilerow_test,rowindex_test, unique_smiles_test = read_gnps.get_class_vectors()
print 'classtest', classtest.shape
smile_latent_dict_train = file_handler.load_latent_dict('massbank_smiles_2_latent.csv')
print 'latent_train', len(smile_latent_dict_train)
smile_latent_dict_test = file_handler.load_latent_dict('gnps_smiles_2_latent.csv')
print 'latent_test', len(smile_latent_dict_test)

#%%
mz_model_dict = {}
smile_binary_true ={}
smile_binary_predict ={}
smile_confusion = {}
smile_sen_spec=[]
sorted_sen_spec=[]


        
# the input is the latent_vectors(2d) and the class vector with each column being the class label
def make_fitted_model(latent_reps, class_vector):
    model = svm.SVC(kernel='rbf',probability=True,C=0.1,gamma=10,class_weight='balanced')
    fitted_model= model.fit(latent_reps,class_vector)
    return fitted_model

#  get a list of smiles that were not used for trainning and return a dictionary
def get_unique_smiles():
    unique_smiles = {}
    for key in unique_smiles_test.keys():
        if unique_smiles_train.get(key) == None:
            unique_smiles[key] = 1
    return unique_smiles

# input one mz_value at a time to get one model, for mz_value with high auc sore
def make_model_dict(mz_value):
    latent_matrix, class_vector = latent_n_class.get_latent_class(classtrain,mz_value_train,mz_value,rowindex_train, smile_latent_dict_train)
    fitted_model = make_fitted_model(latent_matrix, class_vector)
    pickled_model = pickle.dumps(fitted_model)
    mz_model_dict[mz_value] = pickled_model

# get binary vectors for each smile_string, to prepare for the confusion matrix
def get_binary_vector(smile_string):
    temp_smile_binary_true = []
    temp_smile_binary_predict = []
    for mz_value in mz_value_train.keys():
        if mz_value_test.get(mz_value) == None or mz_value_test.get(mz_value) == 0:
            temp_smile_binary_true.append(0)
        else:
            colindex = mz_value_test.get(mz_value)
            print colindex, 'colindex'
            rowindex = smilerow_test.get(smile_string)
            print rowindex, 'rowindex'
            temp_smile_binary_true.append(classtest[rowindex,colindex])
        temp_model = mz_model_dict.get(mz_value)
        unpickled_model = pickle.loads(temp_model)
        if unpickled_model == None:
          print '==============='
          continue
        else:
          latent_reps = smile_latent_dict_test.get(smile_string)
          predicted_value = unpickled_model.predict(latent_reps)
          print predicted_value
          temp_smile_binary_predict.append(predicted_value)
    smile_binary_true[smile_string] = temp_smile_binary_true
    smile_binary_predict[smile_string] = temp_smile_binary_predict
    
def get_confusion_matrix(smile_string):
    temp_true_vector = smile_binary_true.get(smile_string)
    temp_predict_vector = smile_binary_predict.get(smile_string)
    temp_confusion_matrix = confusion_matrix(temp_true_vector,temp_predict_vector)
    smile_confusion[smile_string] = temp_confusion_matrix

def sensitivity_specificity(smile_string):
    temp_confusion_matrix = smile_confusion.get(smile_string)
    sensitivity = temp_confusion_matrix[0,0]/(temp_confusion_matrix[0,0]+temp_confusion_matrix[0,1])
    specificity = temp_confusion_matrix[1,1]/(temp_confusion_matrix[1,0]+temp_confusion_matrix[1,1])
    temp_list = [smile_string, sensitivity, specificity]
    smile_sen_spec.append(temp_list)

def sort_list():
  global sorted_sen_spec
  sorted_sen_spec = sorted(smile_sen_spec, key = operator.itemgetter(1, 2))


def make_mz_model(result_csv_file, num_of_molecule, num_of_positive, auc_score):
  mz_values = mz_value_list.get_mz_list(result_csv_file, num_of_molecule, num_of_positive, auc_score)
  print 'mz_values', len(mz_values)
  for mz_value in mz_values:
#  for index in range (20):
#    mz_value = mz_values[index]
#    print type(mz_value),mz_value
    make_model_dict(mz_value)
#    print index

#%%


make_mz_model('massbank_result.txt', 20, 10,0.9)
#
unique_smiles_all = get_unique_smiles()
print len(unique_smiles_all)
####
for index in range (1):
    smile_string = unique_smiles_all.keys()[index]
    print 'smiles_string', smile_string
    get_binary_vector(smile_string)
    print 'smile-binary-true', len(smile_binary_true)
    print 'smile_binary-predict', len(smile_binary_predict)
###    get_confusion_matrix(smile_string)
#    sensitivity_specificity(smile_string)

#sorted_sen_spec = sort_list()

print type(mz_model_dict), len(mz_model_dict), ',,,,,,,,,,,,'
#%%
for k,val in mz_model_dict.iteritems():
  print type(val)
