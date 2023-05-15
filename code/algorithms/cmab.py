import constants
import constants
import numpy as np
import math
import util

class CMAB:
    
    def __init__(self, u_cc_instance):
        self.u_cc_instance = u_cc_instance
        self.M = u_cc_instance.get_M()
        self.rewards = None
        self.rewards_avg_run = None
        self.rewards_optimal = None
        self.rewards_otimal_avg_run = None
        self.best_clustering = None
        self.best_loss = float("inf")
        self.sample_plus = None
        self.sample_minus = None
        self.oracle = None
        self.num_trials_pivot = None

        self.optimal_clustering = []
        self.optimal_same_cluster_mask = None
        self.optimal_different_cluster_mask = None
        self.optimal_loss = None

    def run(self, T, oracle="pivot", delta_pivot = 1, parallel_execution=True, seed=None, optimal_clustering=None, A=None, log_run_file=None, id_run=1):
        self.rewards = [None] * T
        self.rewards_expected = [None] * T
        self.rewards_optimal = [None] * T
        self.errors_estimates = [None] * T
        self.n_clusters = [None] * T
        self.avg_size_clusters = [None] * T
        self.rewards_avg_run = np.zeros(T)
        self.rewards_expected_avg_run = np.zeros(T)
        self.rewards_optimal_avg_run = np.zeros(T)
        self.errors_estimates_avg_run = np.zeros(T)
        self.n_clusters_avg = np.zeros(T)
        self.avg_size_clusters_avg = np.zeros(T)
        self.oracle = oracle
        self.num_trials_pivot = math.ceil(math.log(self.u_cc_instance.get_number_nodes(), 1 + delta_pivot))
        if seed:
            np.random.seed(seed)
        if oracle == "charikar":
            if A:
                self.A = A
            else:
                self.A = util.build_A(self.u_cc_instance.get_number_nodes())

        if len(self.optimal_clustering) == 0: # set class variables if not yet defined
            if optimal_clustering:
                self.optimal_clustering = optimal_clustering
                #print(optimal_clustering)
            else:
                if oracle == "pivot":
                    # n_runs = constants.num_iterations_optimal
                    # avg_optimal_loss, best_optimal_loss, best_clustering = util.avg_pivot_run(self.u_cc_instance, n_runs)
                    # self.optimal_clustering = best_clustering
                    # print("Computed optimal value of loss! Avg. % .2f - Best (Nearest to avg.) % .2f" % (avg_optimal_loss, best_optimal_loss))
                    self.optimal_clustering = util.best_pivot_run(self.u_cc_instance, self.u_cc_instance.get_w_plus(), self.u_cc_instance.get_w_minus(), self.num_trials_pivot)
                else:
                    assert oracle == "charikar"
                    self.optimal_clustering = util.best_charikar_run(self.u_cc_instance, self.u_cc_instance.get_w_plus(), self.u_cc_instance.get_w_minus(), self.A, self.num_trials_pivot)
                    print("Computed solution w.r.t. true (hidden) weights")
            self.optimal_loss = self.u_cc_instance.analytically_expected_loss(self.optimal_clustering, self.u_cc_instance.get_w_plus(), self.u_cc_instance.get_w_minus())
            self.optimal_same_cluster_mask = np.array([self.optimal_clustering[edge.source] == self.optimal_clustering[edge.target] for edge in self.u_cc_instance.get_graph().es])
            self.optimal_different_cluster_mask = (np.ones(shape=self.u_cc_instance.get_number_edges()) - self.optimal_same_cluster_mask)
        print("Starting CMAB algorithm run #" + str(id_run))
        # remove dots for efficiency
        select_action = self.select_action
        play_action = self.play_action
        update_estimates = self.update_estimates
        if parallel_execution:
            constants.n_runs_per_bandit = 1 # each thread runs 1 time
        #checkpoints = [T//10, T//5, (3*T)//10, (2*T)//5, T//2, (3*T)//5, (7*T)//10, (4*T)//5, (9*T)//10, T-1]
        for run in range(constants.n_runs_per_bandit):
            # at each run need to reset counters and estimates
            self.init_estimates()
            for t in range(T):          
                A = select_action(t)
                play_action(A, t)
                update_estimates(A, t)
                # print("Round " + str(t) + " completed")
                # if log_run_file: # write sometimes to track completion
                #     if t in checkpoints:
                #         with open(log_run_file, "w") as f:
                #             f.write(str(t) + "\n")

            self.last_clustering = A
            if parallel_execution:
                print("Finished CMAB run #" + str(id_run))
            else:
                print("Finished CMAB run #" + str(run))
            self.rewards_avg_run = self.rewards_avg_run + np.array(self.rewards)
            self.rewards_expected_avg_run = self.rewards_expected_avg_run + np.array(self.rewards_expected)
            self.rewards_optimal_avg_run = self.rewards_optimal_avg_run + np.array(self.rewards_optimal)
            self.errors_estimates_avg_run = self.errors_estimates_avg_run + np.array(self.errors_estimates)
            self.n_clusters_avg = self.n_clusters_avg + np.array(self.n_clusters)
            self.avg_size_clusters_avg = self.avg_size_clusters_avg + np.array(self.avg_size_clusters)
        self.rewards = self.rewards_avg_run / constants.n_runs_per_bandit
        self.rewards_expected = self.rewards_expected_avg_run / constants.n_runs_per_bandit
        self.rewards_optimal = self.rewards_optimal_avg_run/ constants.n_runs_per_bandit
        self.errors_estimates = self.errors_estimates_avg_run / constants.n_runs_per_bandit
        self.n_clusters_avg = self.n_clusters_avg / constants.n_runs_per_bandit
        self.avg_size_clusters_avg / self.avg_size_clusters_avg / constants.n_runs_per_bandit
        best_clustering, best_loss = self.get_best_clustering_info()
        return (best_clustering, best_loss, self.compute_regret(T), self.compute_regret(T, expected=True), self.get_cumulative_avg_losses(), self.get_cumulative_avg_expected_losses(), self.get_errors(), self.n_clusters_avg, self.avg_size_clusters_avg, self.get_expected_losses())

    def init_estimates(self):
        self.igraph_graph = self.u_cc_instance.get_graph()
        n_edges = self.u_cc_instance.get_number_edges()
        self.n_plus = np.zeros(shape=n_edges)
        #self.estimate_p_plus = np.ones(shape=n_edges)
        self.estimate_p_plus = np.zeros(shape=n_edges)
        self.n_minus = np.zeros(shape=n_edges)
        #self.estimate_p_minus = np.ones(shape=n_edges)
        self.estimate_p_minus = np.zeros(shape=n_edges)
        self.sample_plus = np.zeros(shape=n_edges)
        self.sample_minus = np.zeros(shape=n_edges)

    #@abstractmethod
    def select_action(self, t):
        if self.oracle == "pivot":
            return util.best_pivot_run(self.u_cc_instance, self.estimate_p_plus, self.estimate_p_minus, self.num_trials_pivot)
        return util.best_charikar_run(self.u_cc_instance, self.estimate_p_plus, self.estimate_p_minus, self.num_trials_pivot)

    def play_action(self, action, t):
        # we sample anyway from all distributions in order to be able to compute the regret component regarding the optimal action
        self.sample_plus, self.sample_minus =  self.u_cc_instance.sample_graph_all()
    
    def update_estimates(self, clustering, t):
        """Method for updating the mean estimates after observing a possible world of interaction
        
        Arguments:
            clustering {list of integer id} -- clustering applied 
            observed_graph {igraph object} -- observed graph with strengths of interaction
        """
        igraph_graph = self.u_cc_instance.get_graph()
        optimal_same_cluster_mask = self.optimal_same_cluster_mask
        optimal_different_cluster_mask = self.optimal_different_cluster_mask
        n_edges = self.u_cc_instance.get_number_edges()
        M = self.M

        same_cluster_mask = np.array([clustering[edge.source] == clustering[edge.target] for edge in igraph_graph.es])
        different_cluster_mask = (np.ones(shape=n_edges) - same_cluster_mask)
        
        self.n_plus = self.n_plus + different_cluster_mask
        self.n_minus = self.n_minus + same_cluster_mask

        scaled_sample_plus = self.sample_plus/M
        scaled_sample_minus = self.sample_minus/M

        ones = np.ones(n_edges)
        self.estimate_p_plus = self.estimate_p_plus + different_cluster_mask * np.divide(ones, self.n_plus, out=np.zeros_like(ones), where=self.n_plus!=0) * (scaled_sample_plus - self.estimate_p_plus)
        self.estimate_p_minus = self.estimate_p_minus + same_cluster_mask * np.divide(ones, self.n_minus, out=np.zeros_like(ones), where=self.n_minus!=0) * (scaled_sample_minus - self.estimate_p_minus)

        cumulative_interaction = np.sum(same_cluster_mask * self.sample_minus + different_cluster_mask * self.sample_plus)
        cumulative_interaction_optimal = np.sum(optimal_same_cluster_mask * self.sample_minus + optimal_different_cluster_mask * self.sample_plus)
        cumulative_interaction_expected = np.sum(same_cluster_mask * self.u_cc_instance.get_w_minus() + different_cluster_mask * self.u_cc_instance.get_w_plus())

        n_clusters, avg_size_clusters = util.get_clustering_info(clustering)
        self.n_clusters[t] = n_clusters
        self.avg_size_clusters[t] = avg_size_clusters

        self.rewards[t] = cumulative_interaction
        self.rewards_optimal[t] = cumulative_interaction_optimal
        self.rewards_expected[t] = cumulative_interaction_expected
        if t + 1 >= round(len(self.rewards) * 0.9) and cumulative_interaction_expected < self.best_loss:
            self.best_loss = cumulative_interaction_expected
            self.best_clustering = clustering
        self.compute_error(t)
        

    def compute_error(self, t):
        # compute errors of mean estimates
        p_plus =  self.u_cc_instance.get_w_plus()
        p_minus = self.u_cc_instance.get_w_minus()
        M = self.M
        diff_plus =  self.estimate_p_plus * M  - p_plus
        diff_minus = self.estimate_p_minus * M - p_minus
        # relative L2 error
        self.errors_estimates[t] = math.sqrt((np.sum(diff_plus * diff_plus) + np.sum(diff_minus * diff_minus))/(np.sum(p_plus * p_plus) + np.sum(p_minus * p_minus)))

    def get_rewards(self):
        return self.rewards

    def get_losses(self):
        return self.rewards
    
    def get_expected_losses(self):
        return self.rewards_expected

    def get_cumulative_avg_losses(self):
        cum_losses = self.get_cumulative_losses()
        T = len(cum_losses)
        len_array = np.arange(1, T+1)
        return cum_losses/len_array
    
    def get_cumulative_avg_expected_losses(self):
        cum_losses = self.get_cumulative_expected_losses()
        T = len(cum_losses)
        len_array = np.arange(1, T+1)
        return cum_losses/len_array

    def get_cumulative_losses(self):
        losses = self.get_losses()
        cum_losses = np.cumsum(losses)
        return cum_losses

    def get_cumulative_expected_losses(self):
        losses = self.get_expected_losses()
        cum_losses = np.cumsum(losses)
        return cum_losses

    def get_best_clustering_info(self):
        return self.best_clustering, self.best_loss

    def get_optimal_action(self):
        return self.optimal_clustering
    
    def get_estimates_plus(self):
        return self.estimate_p_plus
    
    def get_estimates_minus(self):
        return self.estimate_p_minus
    
    def get_nplus(self):
        return self.n_plus
    
    def get_nminus(self):
        return self.n_minus

    def get_errors(self):
        return self.errors_estimates

    def compute_regret(self, T, average=False, expected=False):
        if expected:
            regret_values = self.rewards_expected - np.full_like(self.rewards_expected, fill_value=self.optimal_loss)
        else:
            regret_values = self.rewards - self.rewards_optimal
        # replace negative values with zero --> bad superarm def.
        regret_values[regret_values < 0] = 0
        regret_values = np.cumsum(regret_values)
        if average:
            len_array = np.arange(1, T+1)
            return regret_values/len_array
        return regret_values

