import numpy as np
import matplotlib.pyplot as plt
import math

# TODO: Show training and testing in the same figure
def draw_decision_region(X,y,clf_list,err_list):
    h = 0.01  # step size in the mesh
    y = y.dot(np.array([0,1,2]))
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    for i in range(len(clf_list)):
        clf = clf_list[i]
        print('\n Drawing'+clf.get_title()+'...')
        plt.subplot(2, int(len(clf_list)/2), i + 1)
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)

        result = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = result.argmax(axis=1)
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Set3, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1],s=10 , c=y, edgecolors='#666777', cmap=plt.cm.Set3)
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(clf.get_title()+' (err:%.3f)' % err_list[i])

    plt.show()
