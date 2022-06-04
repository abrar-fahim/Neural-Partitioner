
import matplotlib.pyplot as plt


import time


def get_test_accuracy(model_forest, knn, X_test, k, batch_size=1024, bin_count_param=1, models_path=None):

    

    if models_path == None:
        print('no file directory for models found')
        return
    # assert isinstance(root_model, Model)


    inference_list = torch.empty((0), device='cuda', dtype=int)


    # n = X.shape[0]  # total no of points in dataset

    # FIND CANDIDATE SET OF QUERY POINT START
    start_time = time.time()
    


    print('-----DOING MODEL INFERENCE ------- ')



    query_bins, scores, dataset_bins = model_forest.infer(X_test, batch_size, bin_count_param, models_path)


    

    n_q = query_bins.shape[1] # no of points in test set only

    all_points_bins = []

    ensemble_accuracies = [] # array of accuracies
    ensemble_cand_set_sizes = []

    single_model = model_forest.trees[0].root.model

    print('no of parameterss in one model: {}'.format(sum(p.numel() for p in single_model.parameters())))

    n_trees = model_forest.n_trees

    del model_forest

    torch.cuda.empty_cache()
        



    for num_models in range(n_trees):
    # for num_models in range(n_trees- 1, n_trees): # TAKING ALL TREES AT ONCE

        accuracies = []
        
        # X = list(range(n_bins))
        X = []

    
        for bin_count in range(1, bin_count_param + 1, 1):

         
        
            num_knns = torch.randn(n_q, 1)
            candidate_set_sizes = torch.randn(n_q, 1)

            c1_time = time.time()

            t1_time = c1_time - start_time

            running_time = t1_time


            print("%d models, %d bins "%(num_models + 1, bin_count))
            print()

            max_is = []

            for point in range(n_q):
                c2_time = time.time()

                print('\rpoint ' + str(point) + ' / ' + str(n_q), end='')

                max_val = -1
                max_i = -1
               

                max_i = torch.argmax(scores[:(num_models+1)], 0)[point].flatten()
                # print('max i: {}'.format(max_i))

                max_is.append(max_i)
                # print('query bins shape ', query_bins.shape)
                assigned_bins = query_bins[max_i, point, :].flatten()



                # print('assigned bins ', assigned_bins)
                # print('inf list max i ', inference_list[max_i])

                all_points_bins.append(assigned_bins[0].item())

                candidate_set_points = sum(dataset_bins[max_i].flatten() == b for b in assigned_bins[:bin_count]).nonzero(as_tuple=False).flatten()




                c3_time = time.time()
                t2_time = c3_time - c2_time
                running_time += t2_time

                # FIND CANDIDATE SET OF QUERY POINT END
                candidate_set_size = candidate_set_points.shape[0]
                knn_points = knn[point][:k] # choose first k points for testing
                knn_points_size = knn_points.shape[0]


                # find size of overlap between knn_points and bin_points

                
                knn_and_bin_points = torch.cat((candidate_set_points.cuda(), knn_points.cuda()))
                
                uniques = torch.unique(knn_and_bin_points)

                uniques_size = uniques.shape[0]

                overlap = candidate_set_size + knn_points_size - uniques_size

                # print('overlap ', overlap)

                num_knns[point] = overlap
                
                candidate_set_sizes[point] = candidate_set_size
           
                
                
            pass

            

            accuracy = num_knns / k
            print()

            min_index = torch.argmin(accuracy)

            accuracy = torch.mean(accuracy)

            print('mean acc ', accuracy)
            candidate_set_size = torch.mean(candidate_set_sizes)
            print("candidate set size", candidate_set_size)

            accuracies.append(accuracy.item()) # for each bin_count

            X.append(candidate_set_size.item())
        pass
        ensemble_accuracies.append(accuracies)
        ensemble_cand_set_sizes.append(X)



    if bin_count_param > 0:

        print("first bin accuracy")
        print(accuracies[0])

        print("candidate_set_size of first bin on average")
        print(X[0])
        # print(accuracies[1])
        print(X, accuracies)
        print(ensemble_cand_set_sizes, ensemble_accuracies)

        for m, acc in enumerate(ensemble_accuracies):
            
            plt.plot(ensemble_cand_set_sizes[m], acc, label="no of models: " + str(m+1))
        plt.legend()
        plt.show()

    return all_points_bins, ensemble_cand_set_sizes, ensemble_accuracies
