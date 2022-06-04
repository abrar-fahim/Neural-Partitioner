import torch
from modeltree import ModelTree


class ModelForest():
    def __init__(self, n_trees, n_input, branching, levels, n_classes, model_type='neural') -> None:
        self.trees = [] # list of modeltrees
        self.n_trees = n_trees  
        self.n_input = n_input
        self.branching = branching    
        self.levels = levels
        self.n_classes = n_classes
        self.model_type=model_type
        pass

    def eval(self):
        for (i, model_tree) in enumerate(self.trees):
            model_tree.eval()


    def build_forest(self, load_from_file=False, data=None, models_path=None):
        if models_path == None:
            print('no file directory for models found')
            return
        for i in range(self.n_trees):

            # if i <=0 :
            #     load_from_file = True
            # else:
            #     load_from_file = False
            print('BUILDING TREE %d / %d' % (i, self.n_trees))
            tree = ModelTree(self.n_input, self.branching, self.levels, self.n_classes, model_type=self.model_type) 
            tree.build_tree_from_root(load_from_file=load_from_file, data=data, models_path=models_path)
            self.trees.append(tree)
            self.data = data
        self.load_from_file = load_from_file

    def train_forest(self, X, Y, crit, n_epochs, batch_size, lr, data, models_path):
        n = X.shape[0]

        self.n_points = n
        input_weights = torch.ones(n, device='cuda', requires_grad=False)

        input_weights_sum = torch.sum(input_weights)
        print('old input weights: {}'.format(input_weights))
        old_n_epochs = n_epochs
        
        for (i, model_tree) in enumerate(self.trees):
            
            print('\rtraining model ', i, ' / ', len(self.trees), end='')
            print()
            print('preparing input weights')
            
            running_current_weights = torch.empty((0), device='cuda', requires_grad=False)  # empty tensor to store running current_weights

            # if i <= 0:
            if i <= 0:
                # n_epochs = 0 # SKIPPING 0TH MODEL FOR NOW
                pass
            else:
                n_epochs = old_n_epochs
            for j in range(i):  # all models from 0 to (i-1)
            # for j in range(max(i-1, 0), i):  # only the prev model, except for 0th model

                if n_epochs == 0:
                    break

                print('\rmodel ', j, ' / ', i, end='')
                print()

                running_current_weights = torch.empty((0), device='cuda', requires_grad=False)  # empty tensor to store running current_weights

                model = self.trees[j]
                model.root.to('cuda')
                for k in range(0, n, batch_size):
                    print('\rbatch ', k / batch_size, ' / ', n / batch_size, end='')

                    
                    # start = k * batch_size
                    # end = (k + 1) * batch_size
                    # X_batch = X[start:end]
                    # Y_batch = Y[start:end]

                    X_batch = X[k: k + batch_size]
                    Y_batch = Y[k: k + batch_size]

                    input_weights_batch = input_weights[k: k + batch_size]
                    # print('X batch shape', X_batch.shape)
                    # processing batch
                    knn_indices = torch.flatten(Y_batch)
                    knn_indices = torch.unique(knn_indices)
                    knn_indices, _ = torch.sort(knn_indices, descending=False)
                    knn_indices = knn_indices.type(dtype=torch.long)
                    # print('knns indices', knn_indices)
                    knns = torch.index_select(X, 0, knn_indices)
                    map_vector = torch.zeros(n, device='cuda', dtype=torch.long)
                    nks_vector = torch.arange(knn_indices.shape[0], device='cuda', dtype=torch.long)
                    map_vector = torch.scatter(map_vector, 0, knn_indices, nks_vector)
                    del nks_vector
                    
                    del knn_indices
                    map_vector = map_vector + X_batch.shape[0]
                    X_knn_batch = torch.cat((X_batch, knns), 0)
                    del X_batch
                    
                    del knns
                    
                    y_pred_assigned_bins, scores = model.infer(X_knn_batch, models_path=models_path)
                    
                    Y_batch_shape = Y_batch.shape
                    Y_batch = torch.flatten(Y_batch)
                    Y_batch = Y_batch.type(dtype=torch.long)
                    Y_batch = torch.gather(map_vector, 0, Y_batch) 
                    del map_vector
                    Y_batch = torch.reshape(Y_batch, Y_batch_shape) 
                    Y_batch = Y_batch.type(dtype=torch.double)

                    (_, _, _, _, current_weights) = crit(y_pred_assigned_bins.flatten(), Y_batch, input_weights_batch, calculate_booster_weights=True, n_bins=self.n_classes, confidence_scores=scores)
                    del Y_batch
                    del y_pred_assigned_bins
                    del scores
                    # here, current_weights is a vector of size batch_size
                    current_weights = current_weights.detach() 
                    running_current_weights = torch.cat((running_current_weights, current_weights), 0)
                    # print('cw shape: {}'.format(current_weights.shape))
                    # print('running length', running_current_weights.shape)
                    del current_weights
                    del input_weights_batch

                pass

                model.root.to('cpu')    
                # y_pred = model_list[j](X)
                # (_, _, _, current_weights) = crit(y_pred, Y, input_weights) 
                # current_weights = current_weights.detach()

                # print('iw shape: {}, rcw shape: {}'.format(input_weights.shape, running_current_weights.shape))
                # input_weights *= running_current_weights    # keeping running product of all input weights for all models 0 to (i-1) in input_weights
                input_weights *= running_current_weights    # old models weights are captured in input_weights thru loss fn



            print('weights quartiles: ', torch.quantile(input_weights, torch.tensor([0.25, 0.5, 0.75, 0.8, 0.95, 1], device='cuda')))
            

            # clamp weights to ignore weights above 75th percentile.
            # input_weights = torch.clamp(input_weights, min=0.5, max=torch.quantile(input_weights, 0.9))
            input_weights = torch.clamp(input_weights, min=1e-9)

            
            # make new input weights add up to all ones input weights sum
            input_weights = (input_weights / torch.sum(input_weights)) * input_weights_sum

            print('new input weights: {}'.format(input_weights))

            print('input weights: MAX: {}, MIN: {} '.format(torch.max(input_weights), torch.min(input_weights)))



            torch.cuda.empty_cache()
            print('training')

            

            
            model_tree.train_model_tree_or_infer(X, Y, None, input_weights, crit, data, train_mode=True, batch_size=batch_size,iterations=n_epochs, lr=lr, model_loaded_from_file=self.load_from_file, test_X=None,test_Y=None, tree_index=i, models_path=models_path)

            # del input_weights
            del running_current_weights
            torch.cuda.empty_cache()


    def get_first_model_assigned_bins(self):
        return self.trees[0].assigned_bins


    def get_all_models_assigned_bins(self):
        assigned_bins = []
        for tree in self.trees:
            assigned_bins.append(tree.assigned_bins)
        return assigned_bins

    def get_all_models_bins_data_structures(self):
        bins_data_structures = []
        for tree in self.trees:
            bins_data_structures.append(tree.bins_data_structures)
        return bins_data_structures

    def infer(self, Q, batch_size, bin_count_param, models_path):

        # print(' ----- WHYYYYYYYYYYYYY  2111111------- ')



        query_bins = torch.empty((self.n_trees, Q.shape[0], bin_count_param), device='cuda')

        scores = torch.empty((self.n_trees, Q.shape[0]), device='cuda')

        dataset_bins = torch.empty((self.n_trees, self.n_points, 1), device='cuda')


        for (i, model_tree) in enumerate(self.trees):

            model_query_bins, model_scores = model_tree.train_model_tree_or_infer(None, None, Q, None, None, None, train_mode=False, batch_size=batch_size, bin_count=bin_count_param, models_path=models_path)

            query_bins[i] =  model_query_bins.to('cuda')
            scores[i] = model_scores.to('cuda')
            dataset_bins[i] = model_tree.assigned_bins.to('cuda')

            # inference: (n_models, n, bin_count_param)

            # scores = (n_models, n, 1) -> tells score of first bin only

            # return query inference list of shape (n_models, n_q, bin_count_param)),
            # return inference of dataset points of that model (n_models, n_x, 1)
            # also return scores (n_models, n_x, 1)
            # print(' ----- WHYYYYYYYYYYYYY seogihdioaghsdrsoi;ghdr;sogi------- ')

        # print(' ----- WHYYYYYYYYYYYYY ------- ')
        print('forest inf shapes ', query_bins.shape, scores.shape, dataset_bins.shape)

        return query_bins, scores, dataset_bins





            










        

            
    



    

    