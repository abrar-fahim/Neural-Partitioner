from matplotlib import pyplot as plt
import torch
from treenode import TreeNode



class ModelTree():
    def __init__(self, n_input, branching, levels, n_classes, n_hidden_params, model_type='neural'):
        
        if levels > 0:
            root_classes = branching
        else:
            # self.root.model.__init__(self.n_input, -1, self.n_global_classes, None) 
            root_classes = n_classes
        
        self.root = TreeNode(n_input, n_hidden_params, root_classes, model_type) 

        self.n_hidden_params = n_hidden_params
        self.n_input = n_input
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.primary_device="cuda:0" if torch.cuda.is_available() else "cpu"
        # self.secondary_device="cpu"
        self.secondary_device="cpu" # for small dataset

        self.n_global_classes = n_classes

        self.branching = branching
        self.levels = levels

        self.assigned_bins = None

        self.bins_data_structure = None # shape (n_bins, max_bin_size)
        self.weights = None

        self.model_type = model_type

    pass

    def eval(self):
        self.root.model.eval()

    def train(self):
        self.root.model.train()

    
    def build_tree_from_root(self, load_from_file=False, data='sift', models_path=None):
        # this should only be called from root_model

        if models_path == None:
            print('no file directory for models found')
            return
        levels = self.levels
        branching = self.branching
        n_classes = self.n_global_classes

        print(" -- BUILDING TREE WITH %d LEVELS AND %d BRANCHING -- " % (levels, branching))
        
        self.n_global_classes = n_classes
        # calling again to adjust n_hidden

        self.root.level = 0
        self.root.level_index = 0
        
        q = []
        q.append(self.root)

        n_leaves = branching ** levels

        n_classes_per_leaf = int(n_classes / n_leaves)

        if load_from_file:
            print('LOADING ROOT MODEL FROM FILE')

            self.root.load_params_from_file(data, models_path)


        
        while(len(q) > 0):
            model = q.pop(0)
            
            if model.level >= levels:

                continue
            for b in range(branching):

                
                if model.level >= levels - 1:
                    # model is leaf
                   
                    child = TreeNode(n_input=self.n_input, n_hidden=self.n_hidden_params, num_class=n_classes_per_leaf, model_type=self.model_type).to(self.secondary_device)
                else:
                    child = TreeNode(n_input=self.n_input, n_hidden=self.n_hidden_params, num_class=self.branching, model_type=self.model_type).to(self.secondary_device)
                child.parent = model
                child.level = model.level + 1
                child.index = b

                child.level_index = model.level_index * branching + b
                model.children_models.append(child)
                if load_from_file:
                    print('LOADING MODEL FROM FILE')

                    child.load_params_from_file(data, models_path)
                q.append(child)

    def infer(self, Q, batch_size=None, models_path=None):
        if models_path == None:
            print('no file directory for models found')
            return
        if batch_size is None:
            batch_size = Q.shape[0]
        return self.train_model_tree_or_infer(None, None, Q, None, None, None, train_mode=False,batch_size=batch_size,iterations=None,lr=None, bin_count=1, model_loaded_from_file=None, test_X=None, test_Y=None, tree_index=None, models_path=models_path)


    def train_model_tree_or_infer(self, X, Y, Q, input_weights, crit, data, train_mode=True,batch_size=4096,iterations=5,lr=1e-3, bin_count=1, model_loaded_from_file=False, test_X=None, test_Y=None, tree_index=0, models_path=None):

        if models_path == None:
            print('no file directory for models found')
            return

        if not train_mode:
            return self.infer_new(Q, batch_size=batch_size, bin_count=bin_count)

        # if train_mode, then trains tree, else runs inference with Q as query set

        leaves = []

        q = [] # FIFO (queue) data structure

        base_iters = iterations


        start = True
        
        q.append(self.root) 

        is_root = True

        if train_mode:
            input = X
        else:
            input = Q

        


        while(len(q) > 0): # BFS
            

            
            model = q.pop(0)
           
            
            
            assert isinstance(model, TreeNode)
            model.to("cuda")

            
            

            # find input points using parent's point distribution
            # X = X # X is the input var here

            if train_mode:
                
                print('\nTRAINING MODEL level : %d, level index: %d / %d'%(model.level, model.level_index, self.branching ** model.level))

                
                # input = X                
            else:
                print('\ninfering on MODEL level : %d, level index: %d / %d'%(model.level, model.level_index, self.branching ** model.level))
                # input = Q

            if model.parent is not None:
                parent_distribution = model.parent.distribution.cuda() # shape(n,), index i contains assigned model of point i

                # print('parent distribution ', parent_distribution)

                # find all points assigned to this model, indexed wrt parent model's input points 

                # print('parent dist: ', parent_distribution)
                # print('model index ', model.index)
                model_input_indices_wrt_parent = (parent_distribution == model.index).nonzero(as_tuple=False).flatten()

                del parent_distribution

                # print('model input indices wrt parten', model_input_indices_wrt_parent)

                model_input_indices_wrt_X = torch.index_select(model.parent.input_indices.cuda(), 0, model_input_indices_wrt_parent)

                del model_input_indices_wrt_parent

                
                

                model_input = torch.index_select(input, 0, model_input_indices_wrt_X)

                # print('model input, ', model_input)
                if train_mode:
                    model_labels = torch.index_select(Y, 0, model_input_indices_wrt_X)


                    nks_vector = torch.arange(model_input.shape[0], device='cuda', dtype=torch.long)

                    # map_vector = torch.zeros(X.shape[0], device='cuda', dtype=torch.long)
                    map_vector = torch.full((X.shape[0],), -1, device='cuda', dtype=torch.long) # there will be -1s where there is no mapping

                    map_vector = torch.scatter(map_vector, 0, model_input_indices_wrt_X, nks_vector)

                    # map_vector[model_input_indices_wrt_X[i]] = nks_vector[i]


                    model_labels_shape = model_labels.shape

                    model_labels = torch.flatten(model_labels)

                    model_labels = model_labels.type(dtype=torch.long)

                    model_labels = torch.gather(map_vector, 0, model_labels)
                    # model_labels[i] = map_vector[model_labels[i]]

                    model_labels = torch.reshape(model_labels, model_labels_shape)

                    model_labels = model_labels.type(dtype=torch.double)

                    model_label_invalid = (model_labels.flatten() == -1).nonzero(as_tuple=False).flatten()

                    print('-1s present: {}'.format(model_label_invalid.shape[0]))

                    # print('model labels after, ', model_labels[:10].type(dtype=torch.long))

                    # model_labels[i] = model_inputs[]



                    # NEED TO REPLACE VALUES IN MODEL_LABELS WITH CORRESPONDING INDICES IN MODEL_INPUT

                    # for value y in Y, 

                model.input_indices = model_input_indices_wrt_X.to(self.secondary_device)
                del model_input_indices_wrt_X


            else:
                model_input = input
                model_labels = Y

                model.input_indices = torch.arange(0, input.shape[0], dtype=int, device=self.secondary_device)

            # print('model input indices for model level %d, level index %d: \n'%(model.level, model.level_index), model.input_indices)

            # print('inputs ', model.input_indices)
            # first train this model

            model_input_weights = torch.index_select(input_weights.to(self.secondary_device), 0, model.input_indices)

            model_input_weights = model_input_weights.to(self.primary_device)

            print('model loadedd from file ', model_loaded_from_file)
            
            # if train_mode and not model_loaded_from_file:
            if train_mode:
                model.train()


                dynamic_iters = (model.level + 1) * base_iters 
   
               
                
                use_generated_batches = False
                use_new_knn_matrix = False

                if model.parent is not None:
                    use_generated_batches = True
                    # use_new_knn_matrix = True
                (losses, acc_losses) = model.train_model(model_input, model_labels, model_input_weights, tree_index, crit, data, batch_size=batch_size, iterations=dynamic_iters, lr=lr,use_generated_batches=use_generated_batches, use_new_knn_matrix=use_new_knn_matrix, models_path=models_path) 

                

                del model_labels

                
                if start:
                    # show loss plot for root model only
                    plt.plot(losses)
                    plt.title('Total Loss vs Epoch')
                    plt.show()
                    # plt.plot(acc_losses, label='add sums')
                    # plt.show()
                
                    start = False

                



            model.eval()


            with torch.no_grad():



                outputs = torch.empty(0, device='cuda')
                n_x = model_input.shape[0]

                for k in range(0, n_x, batch_size):

                    model_input_batch = model_input[k: k + batch_size]
                    batch_outputs = model.infer(model_input_batch)
                    del model_input_batch
                    batch_outputs = batch_outputs.detach()


                    batch_outputs = torch.argmax(batch_outputs, dim=1)
                    outputs = torch.cat((outputs, batch_outputs), dim=0)
                    del batch_outputs

                del model_input
                model.distribution = outputs


                uniques = torch.unique(model.distribution.flatten(), return_counts=True)[1].float()
                cand_set_size = torch.max(uniques)

                model.distribution = model.distribution.to(self.secondary_device)
                del outputs
                del cand_set_size



                # add children to q


                model.to(self.secondary_device)

                

                

                for (i, child) in enumerate(model.children_models):
                    q.append(child)

                if len(model.children_models) == 0:
                    leaves.append(model)

            
            
            is_root = False
           

            
            torch.cuda.empty_cache()

        # if training, generate assigned bins, else generate output for Q
        n = input.shape[0]

       
        
        


        torch.cuda.empty_cache()

        # preparing bins data structure
        Q_assigned_bins = None
        if train_mode:

            self.assigned_bins = torch.empty((n, 1), device=self.secondary_device, dtype=int)  # doesn't need to be a tensor or in cuda
        else:
            Q_assigned_bins = torch.empty((n, bin_count), device=self.secondary_device, dtype=int)  # doesn't need to be a tensor or in cuda

        Q_scores = torch.empty((n, 1), device=self.secondary_device)

        n_leaves = self.branching ** self.levels

        n_classes_per_leaf = int(self.n_global_classes / n_leaves)


        for (i, leaf) in enumerate(leaves):

            # print('leaf ', leaf.level_index)
            # leaf_input = torch.index_select(input, 0, leaf.input_indices.cuda())
            leaf_input = input[leaf.input_indices]

            leaf_input = leaf_input.to(self.primary_device)

            # batch leaf input

            bins = torch.empty(0, device='cuda', dtype=torch.long)

            leaf.to(self.primary_device)
            torch.cuda.empty_cache()
            scores = torch.empty(0, device='cuda', dtype=torch.long)
            n_leaf = leaf_input.shape[0]
            for k in range(0, n_leaf, batch_size):
                
                leaf_input_batch = leaf_input[k: k + batch_size]
                # print('leaf input batch shape: {}'.format(leaf_input_batch.shape))

                batch_outputs = leaf.infer(leaf_input_batch)
                del leaf_input_batch
                batch_outputs = batch_outputs.detach()
                batch_bins = torch.topk(batch_outputs, k=bin_count, dim=1, sorted=True, largest=True)[1]

                batch_scores = torch.topk(batch_outputs, k=1, dim=1, sorted=True, largest=True)[0]

                scores = torch.cat((scores, batch_scores), dim=0)


                bins = torch.cat((bins, batch_bins), dim=0)



            
            del leaf_input
            

           
            bins += leaf.level_index * n_classes_per_leaf



            bins = bins.to(self.secondary_device)
            
            scores = scores.to(self.secondary_device)


            
  


            if train_mode:
                self.assigned_bins[leaf.input_indices] = bins
                # print('assigned bins ', self.assigned_bins)
                self.assigned_bins = self.assigned_bins.to(self.secondary_device)
            else:
                
                
                Q_assigned_bins[leaf.input_indices] = bins
                Q_scores[leaf.input_indices] = scores
                # assigning only the most probable bin score (instead of all bin_count) bins' scores
                

            leaf.to(self.secondary_device)


            del bins
       

        

        torch.cuda.empty_cache()

        # create a bins data structure of size (bin_count, max_bin_size)
        # pad with -1s to solve uneven distribution problem



        if train_mode:

            # max bin size 
            unique_bins = torch.unique(self.assigned_bins.flatten(), return_counts=True)[1]
            # print('uniques ', unique_bins)

            max_bin_size = torch.max(unique_bins)

            

            self.bins_data_structure = torch.full((self.n_global_classes, max_bin_size), -1, device=self.primary_device, dtype=torch.long)

            self.assigned_bins = self.assigned_bins.to(self.primary_device)

            # print('bins data struc ', self.bins_data_structure.shape)

            for i in range(self.n_global_classes):

                candidates = (self.assigned_bins == i).nonzero(as_tuple=True)[0]

                bin_size = candidates.shape[0]
                self.bins_data_structure[i, :] = torch.cat((candidates, torch.full((max_bin_size - bin_size,), -1, device=self.primary_device)))

            # print('bins ds', self.bins_data_structure)

            self.assigned_bins = self.assigned_bins.to(self.secondary_device)

            self.bins_data_structure = self.bins_data_structure.to(self.secondary_device)

        torch.cuda.empty_cache()


        return (Q_assigned_bins, Q_scores)


    def infer_new(self, Q, batch_size=4096, bin_count=1):
        # Q shape: (nq, d)

        q = [] # FIFO (queue) data structure

        q.append(self.root)

        nq = Q.shape[0]

        outputs = torch.ones((nq, self.n_global_classes), device=self.primary_device)

        n_leaves = self.n_global_classes

        scores = torch.ones((nq,), device=self.primary_device)

        while(len(q) > 0): # BFS
            model = q.pop(0)
            assert isinstance(model, TreeNode)
            model.to(self.primary_device)
            # print('\ninfering on MODEL level : %d, level index: %d / %d'%(model.level, model.level_index, self.branching ** model.level))

            # make all models infer on all points, then multiply probabilities to get probabilites of assigned bins

            model.eval()

            

            model_branching = self.branching
            with torch.no_grad():
            
                for k in range(0, nq, batch_size):
                    batch_input = Q[k: k + batch_size]

                    batch_outputs, batch_confidence_scores = model.infer(batch_input, return_confidence_scores=True)

                    start_bin = int(model.level_index * n_leaves / (model_branching ** model.level))

                    bin_range = int(n_leaves / (model_branching ** model.level))




                    outputs[k: k + batch_size, start_bin: start_bin + bin_range] *= batch_outputs.repeat_interleave(bin_range // model_branching, dim=1)

                    scores[k: k + batch_size] *= batch_confidence_scores


                        
                    del batch_input
                    del batch_outputs
                    del batch_confidence_scores
                
            model.to(self.secondary_device)

            for (i, child) in enumerate(model.children_models):
                q.append(child)
        
        Q_assigned_bins = torch.topk(outputs, k=bin_count, dim=1, sorted=True, largest=True)[1]

        # scores = torch.topk(outputs, k=1, dim=1, sorted=True, largest=True)[0]

        return (Q_assigned_bins, scores)


            
