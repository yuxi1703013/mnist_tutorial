from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import metrics
X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)
'''
X = X / 255.
img1 = X[1].reshape(28, 28)
plt.imshow(img1, cmap='gray')
plt.show()
img2 = 1 - img1
plt.imshow(img2, cmap='gray')
plt.show()
img3 = img1.transpose()
plt.imshow(img3, cmap='gray')
plt.show()
'''

X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

#Q1：Logistic Regression
print('Logistic Regression:')
classifier1=LogisticRegression()
classifier1.fit(X_train,Y_train)
predict_train1=classifier1.predict(X_train)
predict_test1 = classifier1.predict(X_test)
train_accuracy1=metrics.accuracy_score(Y_train, predict_train1)
test_accuracy1=metrics.accuracy_score(Y_test, predict_test1)
print('Training accuracy: %0.2f%%' % (train_accuracy1*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy1*100))

#Q2:naive bayes
print('naive bayes:')
classifier2=BernoulliNB()
classifier2.fit(X_train, Y_train)
predict_train2=classifier2.predict(X_train)
predict_test2 = classifier2.predict(X_test)
train_accuracy2=metrics.accuracy_score(Y_train, predict_train2)
test_accuracy2=metrics.accuracy_score(Y_test, predict_test2)
print('Training accuracy: %0.2f%%' % (train_accuracy2*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy2*100))

#Q3:SVM (default parameters)
print('SVM(default parameters):')
classifier3=LinearSVC()
classifier3.fit(X_train, Y_train)
predict_train3=classifier3.predict(X_train)
predict_test3 = classifier3.predict(X_test)
train_accuracy3=metrics.accuracy_score(Y_train, predict_train3)
test_accuracy3=metrics.accuracy_score(Y_test, predict_test3)
print('Training accuracy: %0.2f%%' % (train_accuracy3*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy3*100))

#Q4:SVM (change parameters)
print('SVM(change parameters):')
classifier4=LinearSVC(tol=0.001,C=0.8,max_iter=2000)
classifier4.fit(X_train, Y_train)
predict_train4=classifier4.predict(X_train)
predict_test4 = classifier4.predict(X_test)
train_accuracy4=metrics.accuracy_score(Y_train, predict_train4)
test_accuracy4=metrics.accuracy_score(Y_test, predict_test4)
print('Training accuracy: %0.2f%%' % (train_accuracy4*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy4*100))