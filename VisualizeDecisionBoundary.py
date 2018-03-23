class VisualizeDecisionBoundary:
    
    """Visualize the decision boundary. Scatter the input data points on the plotted decision boundary
    
    The classifier `clf` is already fit on the training data. It uses the plt.cm.Paired color map to 
    visualize the decision boundary and data points together on a single graph. The maximum number of classes
    that can be visualized = 6.
    
    """
    
    def __init__(self, X, y, clf):
        self.X = X
        
        # The input data must have 2 features only because the visualization plots two features only
        assert(X.shape[1] == 2)
        self.y = y
        self.n_classes = len(set(y))
        assert(self.n_classes <= 6)
        self.clf = clf
    
    def visualize(self):
        """Plots the decision boundary and then scatter plots the points"""
        
        # Shouldn't need a plot_step smaller than this. So, not configurable
        plot_step = 0.02
        feature_1_min = self.X[:, 0].min()
        feature_1_max = self.X[:, 0].max()

        feature_2_min = self.X[:, 1].min()
        feature_2_max = self.X[:, 1].max()

        xx, yy = np.meshgrid(np.arange(feature_1_min - 1, feature_1_max + 2, plot_step),
                             np.arange(feature_2_min - 1, feature_2_max + 2, plot_step))

        
        # Data grid coordinates flattened to an np array
        data_as_grid = np.asarray(list(zip(xx.ravel(), yy.ravel())))
        
        # Make the predictions on the data grid
        predictions = clf.predict(data_as_grid)
        
        # Reshape the predicitions
        predictions = np.reshape(predictions, xx.shape)
        
        # Plot the decision surface
        fig = plt.figure()
        ax = plt.gca()
        
        # Paired is a qualitative color map
        ax.contourf(xx, yy, predictions, cmap = plt.cm.Paired)
        
        # This retrieved colorbar instance is used to color data points in the scatter plot

        for color_index, this_class in enumerate(range(0, self.n_classes)):
            X_this_class = self.X[np.where(y == this_class)]
            ax.scatter(X_this_class[:, 0], X_this_class[:, 1])
