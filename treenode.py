from matplotlib import pyplot as plt
import torch

from model import LinearModel, NeuralModel

import prepare

from sklearn import preprocessing

class TreeNode():
    def __init__(self,  n_input, n_hidden, num_class, model_type='neural'):



        self.n_hidden =  128

        self.n_hidden = n_hidden


        if model_type == 'neural':

            self.model = NeuralModel(n_input=n_input, n_hidden=self.n_hidden, num_class=num_class, opt=None).cuda()
        elif model_type == 'linear':
            self.model = LinearModel(n_input=n_input, n_hidden=self.n_hidden, num_class=num_class, opt=None).cuda()
        else:
            raise ValueError('model_type must be either neural or linear')

        self.n_classes = num_class

        
        self.parent = None
        self.input_indices = []

        self.parent = None

        self.children_models = []

        self.distribution = None

        self.level = -1
        self.index = -1

        self.id = None
         # level and level_index uniquely identifies a model within a tree

        self.level_index = -1 # used to calculate actual bin index during inference, to order all leaves (and internal nodes) sequentially

        self.scaler = None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def infer(self, X, return_confidence_scores=False):

        # if X.shape[0] > 0:
        X = X.to('cpu')
        X_scaled = self.scaler.transform(X)
        X_scaled = torch.tensor(X_scaled, device='cuda:0')

        inference, confidence_scores = self.model(X_scaled)
        if return_confidence_scores:
            return inference, confidence_scores
        else:
            return inference

    def cuda(self):
        self.model.cuda()
        

    def train_model(self, X, Y, input_weights, ensemble_model_index, crit, data, batch_size=4096, iterations=5, lr=1e-3, use_generated_batches=False, use_new_knn_matrix=False, models_path = None):

        if models_path == None:
            print('no file directory for models found')
            return

       

        print('training model with {} iters and {} lr, {} hidden params and {} classes'.format(iterations, lr, self.n_hidden, self.n_classes))

        print("standardizing data ")

        
        X = X.to('cpu')
        self.scaler = preprocessing.StandardScaler().fit(X)
        # self.scaler = preprocessing.StandardScaler(with_mean=False, with_std=False).fit(X)
        X_scaled = self.scaler.transform(X)

        X_scaled = torch.tensor(X_scaled, device='cuda:0')

        
        losses = []

        acc_losses = []



        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if self.parent is None:
                self.id = str(ensemble_model_index) + str(self.level) + str(self.level_index) 
        else:
            self.id = self.parent.id + str(self.level) + str(self.level_index)

        n = X.shape[0]

        if n == 0:
            return []  # nothing to train

        lowest_loss = float('inf')

        class options(object):
            if data == 'sift':
                normalize_data=False
                sift=True
            else:
                normalize_data=False
                sift=False
                pass
            pass

        if use_new_knn_matrix:
            print('Preparing new k-NN Matrix')
            Y = prepare.dist_rank(X, Y.shape[1], opt=options, data=data).float()

            Y = Y.to('cuda:0')

       
        # use_generated_batches = False

        if use_generated_batches and iterations > 0:
            # GENERATE BATCHES BEFOREHAND

            print('GENERATING BATCHES')
            

            d = X.shape[1]

            

            pass
            n_batches = int(n / batch_size) + 1

            X_batches = torch.empty((n_batches, batch_size, d),device='cuda')
            
            Y_batches = torch.empty((n_batches, batch_size, Y.shape[1]), device='cuda', dtype=float)

            input_weights_batches = torch.empty((n_batches, batch_size), device='cuda')


            for k in range(int(n / batch_size) + 1):
                random_indices = torch.randint(0, n, (batch_size,), device='cuda')
                X_batch = torch.index_select(X_scaled, 0, random_indices)
                input_weights_batch = torch.index_select(input_weights, 0, random_indices)

                # GENERATE Y_BATCH
                Y_batch = prepare.dist_rank(X_batch, Y.shape[1], opt=options, data=data).float()

                X_batches[k] = X_batch
                Y_batches[k] = Y_batch
                input_weights_batches[k] = input_weights_batch
                del input_weights_batch

          

        for ep in range(1, iterations + 1):


            
            

            loss = 0
            print("TRAINING")

            if batch_size < n:
                print('\nepoch ', ep, ' / ', iterations)
            else:
                print('\repoch ', ep, ' / ', iterations, end='')
            loss_sum = 0


            add_sum = 0
            b_sum = 0


            for k in range(int(n / batch_size) + 1):
                

                optimizer.zero_grad()

                

                if batch_size < n and not use_generated_batches:
                    # print()

                    
                    print('\rtraining batch ', k, '/ ', n / batch_size, end='')
                    # print('BATCHH')

                    collect_knns = True # = True for better model performance

                    # randomly sampling batch_size data points to create X_batch corresponding Y_batch

                    random_indices = torch.randint(0, n, (batch_size,), device='cuda')



                    X_batch = torch.index_select(X_scaled, 0, random_indices)
                    Y_batch = torch.index_select(Y, 0, random_indices)

                    input_weights_batch = torch.index_select(input_weights, 0, random_indices)

                    del random_indices

                    if collect_knns:


                        knn_indices = torch.flatten(Y_batch)  # knn_indices contains indices of knns of all points in X_batch
                        knn_indices = torch.unique(knn_indices)
                        knn_indices, _ = torch.sort(knn_indices, descending=False) # sorting needed to make deterministic the order in which knns are present in (nks, d) shaped knns matrix
                        knn_indices = knn_indices.type(dtype=torch.long)
                        # print('knns indices', knn_indices)

                        # temporarily pad X with nan and make knn_indices refer to this if index out of bounds
                        

                        # keep only indices that in range of this model's X
                        knn_indices = knn_indices[(knn_indices < X.shape[0]) & (knn_indices >= 0)]

                        

                        knns = torch.index_select(X_scaled, 0, knn_indices)



                        map_vector = torch.zeros(n, device='cuda', dtype=torch.long) # maps from X index to index in knns

                        nks_vector = torch.arange(knn_indices.shape[0], device='cuda', dtype=torch.long) # just a vector containing numbers from 0 to nks - 1

                        map_vector = torch.scatter(map_vector, 0, knn_indices, nks_vector)  # map vector shape should be (n)

                        # map_vector[knn_indices[i]] = nks_vector[i]

                        #position of point i in knns vector is map_vector[i]
                        # need to offset these positions by ns since map_vector is contatenated to X_batch, whose size is ns

                        map_vector = map_vector + X_batch.shape[0]



                        X_knn_batch = torch.cat((X_batch, knns), 0)
                        
                        
                        del knns


                        y_pred, confidence_scores = self.model(X_knn_batch)
                        
                        # del X_batch

                        # replace values in Y_batch with corresponding indexes from map_vector
                        
                        Y_batch_shape = Y_batch.shape
                        Y_batch = torch.flatten(Y_batch)
                        Y_batch = Y_batch.type(dtype=torch.long)




                        # next 2 lines replace -1s in y_batch to point to last value in map_vector, which is -1. Basically keeping -1s in Y_batch to be handled later and not get ScatterGatherKernel CUDA error in torch gather
                        map_vector = torch.cat((map_vector, torch.tensor([-1], device='cuda')), dim=0)
                        Y_batch = torch.where(Y_batch < 0, map_vector.shape[0] - 1, Y_batch)


                        Y_batch = torch.gather(map_vector, 0, Y_batch)  # y_batch[i] = map_vector[y_batch[i]]

                        


                        Y_batch = torch.reshape(Y_batch, Y_batch_shape) 
                        Y_batch = Y_batch.type(dtype=torch.double)  # since trunc in loss fn isnt implemented for int or long dtypes


                        # c = torch.zeros(batch_size)
                        c = torch.zeros(X_knn_batch.shape[0])
                        s = torch.zeros(X_knn_batch.shape[0])


                        c[:batch_size] = 0
                        c[batch_size:] = 1 
                        s[:batch_size] = 10
                        s[batch_size:] = 1 



                        
                    else:
                        
                        y_pred, confidence_scores = self.model(X_batch)
                    
                else:
                    # -- NO BATCHING --

                    if not use_generated_batches:

                        X_batch = X_scaled
                        Y_batch = Y
                        input_weights_batch = input_weights
                    else:
                        print('\rtraining batch ', k, '/ ', n / batch_size, end='')
                        rand = torch.randint(X_batches.shape[0], (1,), device='cuda').flatten()[0]
                        X_batch = X_batches[rand]
                        Y_batch = Y_batches[rand]
                        input_weights_batch = input_weights_batches[rand]

                    y_pred, confidence_scores = self.model(X_batch)

                    if ep == iterations:
                        print('y pred ', y_pred)
                    
                n_bins = y_pred.shape[1]
                # print('y pred shape before ', y_pred.shape)

                
                zeros = torch.zeros(1, n_bins, device='cuda')
                Y_batch = torch.where((Y_batch < y_pred.shape[0]) & (Y_batch >= 0), Y_batch, float(y_pred.shape[0]))

                y_pred = torch.cat((y_pred, zeros), dim=0)

                del X_batch
                

                del zeros

                
                running_b_top_size_bound = batch_size
                (loss, diff_sum, b, conf_loss, _ ) = crit(y_pred, Y_batch, input_weights_batch, org_n=running_b_top_size_bound, confidence_scores=confidence_scores)
                del Y_batch
                
                del y_pred

                del confidence_scores

                loss.backward() # only backwarding here, stepping each epoch

                optimizer.step()


                loss_sum += loss.detach()

                b_sum += b.detach()

                add_sum += diff_sum.detach()

                del diff_sum

                del b
                del loss
                del conf_loss


                if batch_size >= n:
                    break # break out of for loop
                torch.cuda.empty_cache()
                                
            pass


            
            n_batches = int(n / batch_size) + 1

            if batch_size >= n:
                n_batches = 1

            print()

            acc_losses.append((add_sum.item() / n_batches))
                
        

            print('b sum ', b_sum / n_batches)


        

            if loss_sum < lowest_loss:
                lowest_loss = loss_sum
                torch.save(self.model.state_dict(), models_path + '/' + data + '-best-model-' + self.id + '-parameters.pt')
            
            pass

        
            n_batches = int(n / batch_size) + 1
            losses.append(loss_sum.item() / n_batches)

            if loss_sum == 0:
                print('loss is 0, BREAKING')
                break
            del loss_sum
        # loading best version of model after training

        # print('id ', self.id)
        self.model.load_state_dict(torch.load(models_path + '/' + data + '-best-model-' + self.id + '-parameters.pt'))
        print()
       

        print()

        self.to('cuda:0')
        self.eval()

        return losses, acc_losses


    def load_params_from_file(self, data, models_path):

        if self.parent is None:
                self.id = str(0) + str(self.level) + str(self.level_index) 
        else:
            self.id = self.parent.id + str(self.level) + str(self.level_index)
            
        self.model.load_state_dict(torch.load(models_path + '/' + data + '-best-model-' + self.id + '-parameters.pt'))

        self.to('cuda:0')
        self.eval()


