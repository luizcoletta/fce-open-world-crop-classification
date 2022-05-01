#inserir a definição dos classificadores aqui

from sklearn import svm

class clfiers:

    def __init__(self, model_name, train, train_labels, test, test_labels):

        if model_name == 'svm':
            self.probs, self.pred = self.svmClassification(train,train_labels, test)


    def svmClassification(self, train, train_labels, test):
        SVM = svm.SVC(tol=1.5, probability=True)
        SVM.fit(train, train_labels)
        probs = SVM.predict_proba(test)
        pred = SVM.predict(test)
        # print(np.around(probs,2))
        return [probs, pred]