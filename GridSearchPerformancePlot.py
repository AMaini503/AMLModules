class GridSearchCVPerformancePlot:
    
    """Draws the param vs score graph from a grid object"""
    
    def __init__(self, grid, num_of_plots_per_row = 2):
        """Stores the grid object and configuration for subplot grid"""
        self.grid = grid
        self.num_of_plots_per_row = num_of_plots_per_row
    
    def getScoresFor(self, param_name):
        """Given a param name, returns a list of train and test scores 
        for that param.
        
        The param values are obtained from grid.params list. The
        list is assumed to be sorted. For reassurance, the param_value
        vs score will be sorted by param_value at the end.
        
        There is one-to-one mapping between the scores and the params
        list
        
        
        """
        # A dictionary that holds scores for a given param_value.
        param_value_to_train_score = dict()
        param_value_to_test_score = dict()
        
        for index, param_dict in enumerate(self.grid.cv_results_['params']):
            # param_dict is of form: 
            # {'param_1': x, 'param_2': y, 'param_3': z ...}
            value_for_this_param = param_dict[param_name]
            
            # If this value has not been seen before, assign a new 
            # list for this value
            if(value_for_this_param not in param_value_to_train_score):
                param_value_to_train_score[value_for_this_param] = []
                param_value_to_test_score[value_for_this_param]  = []
            
            # Append this score to the list of scores for this value
            train_score = grid.cv_results_['mean_train_score'][index]
            test_score = grid.cv_results_['mean_test_score'][index]
            param_value_to_train_score[value_for_this_param].append(train_score)
            param_value_to_test_score[value_for_this_param].append(test_score)
                
        for param_value in param_value_to_train_score:
            
            # Assign the mean of the scores to that param value
            param_value_to_train_score[param_value] = np.mean(
                                    param_value_to_train_score[param_value])
            
            param_value_to_test_score[param_value] = np.mean(
                                    param_value_to_test_score[param_value])
        
        # Sort the train_scores by the param_value
        # First element in the tuple is the key and the second element
        # is the corresponding score
        train_scores_tuples = list(param_value_to_train_score.items())
        train_scores_tuples.sort(key = lambda x: x[0])
        
        test_scores_tuples = list(param_value_to_test_score.items())
        test_scores_tuples.sort(key = lambda x: x[0])
        
        return train_scores_tuples, test_scores_tuples
    
    def getParamNames(self):
        """Gets the names of the parameters from grid.params
        
        Ideally, all the parameter names are present in every element
        of the `grid.params` list
        
        """
        param_names = []
        
        # Gets the first element from the list. Every element has all
        # the parameter names, so, using the first element suffices
        return list(grid.cv_results_['params'][0])
    
    def plot(self):
        """Gets the scores using getScoresFor() and then plots them"""
        param_names = self.getParamNames()
        
        num_of_rows_in_grid = int(np.ceil(len(param_names) / 2))
        fig, axes = plt.subplots(num_of_rows_in_grid, 2, figsize = (20, 5 * num_of_rows_in_grid))
        axes = axes.ravel()
        
        for index, param_name in enumerate(param_names):
            train_scores_tuples, test_scores_tuples = self.getScoresFor(param_name = param_name)
            
            # Plot the train_scores
            param_values_train = [t[0] for t in train_scores_tuples]
            train_scores = [t[1] for t in train_scores_tuples]
            plot1_train_score, = axes[index].plot(param_values_train, train_scores, label = 'mean_training_score')
            
            # Plot the test scores
            param_values_test = [t[0] for t in test_scores_tuples]
            test_scores = [t[1] for t in test_scores_tuples]
            plot2_train_score, = axes[index].plot(param_values_test, test_scores, label = 'mean_test_score')
            axes[index].legend(handles = [plot1_train_score, plot2_train_score])
            
            axes[index].set_xlabel(param_name, fontsize='xx-large')
            axes[index].set_ylabel('Score', fontsize='xx-large')
            axes[index].tick_params(labelsize = 'xx-large')
            
