import pickle
import numpy


class MlPredict():
    def __init__(self, data) -> None:
        '''Predicts using the Random Forest Classification and SVM model'''
        self.data = numpy.array(data).reshape(1, -1)
    
    def predict(self, model_type):
        # Loading the scaler for feature scaling
        scaler = pickle.load(open('scaler.pkl', mode='rb'))
        with open('scaler.pkl', mode='rb') as file:
            scaler = pickle.load(file)
        # Loading ML Models        
        with open('./ML Models/RandomForest_model.pkl', mode='rb') as file_obj:
            rf_model = pickle.load(file_obj)        
        with open('./ML Models/SupportVectorMachine_model.pkl', mode='rb') as f:
            svm_model = pickle.load(f)

        if model_type == 'Random Forest Model (100%)':
            return [rf_model.predict(scaler.transform(self.data))]
        elif len(model_type) == 2:
            return [rf_model.predict(scaler.transform(self.data)), svm_model.predict(scaler.transform(self.data))]
        elif len(model_type) == 0:
            return [rf_model.predict(scaler.transform(self.data))]
        else:
            return [svm_model.predict(scaler.transform(self.data))]


# Testing
if __name__ == '__main__':
    ml_model = MlPredict(data=[59, 1, 1, 140, 221, 0, 1, 164, 1, 0.0, 2, 0, 2])
    print(ml_model.predict())



