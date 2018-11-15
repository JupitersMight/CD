import matplotlib.pyplot as plt

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('original sensitivity', 'original specificity', 'Smote sensitivity', 'Smote specificity')
y_pos = np.arange(len(objects))
performance = [15, 93, 62, 87]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('%')
plt.title('Before and after Smote over sampling with decision tree classifier')
plt.savefig('SMOTE.png', dpi=100)
plt.show()


objects = ('original sensitivity', 'original specificity', 'under sensitivity', 'under specificity')
y_pos = np.arange(len(objects))
performance = [0, 100, 98, 69]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('%')
plt.title('Before and after under sampling with decision tree classifier')
plt.savefig('Under.png', dpi=100)
plt.show()

objects = ('original sensitivity', 'original specificity', 'outliers sensitivity', 'outliers specificity')
y_pos = np.arange(len(objects))
performance = [10, 99, 98, 71]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('%')
plt.title('With and without outliers with KNN')
plt.savefig('outliers.png', dpi=100)
plt.show()

objects = ('original sensitivity', 'original specificity', 'sensitivity normalized', 'specificity normalized')
y_pos = np.arange(len(objects))
performance = [0, 100, 98, 71]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('%')
plt.title('With and without normalization with APS dataset and decision tree classifier')
plt.savefig('normalization.png', dpi=100)
plt.show()