import math
from tqdm import tqdm
from utils import *



class MyLyapunuov:
    def __init__(self, num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V):
        self.num_node = num_node
        self.arrived_lists = arrived_lists
        self.len_T = len_T
        self.Q = [0] * self.len_T
        self.E_avg = E_avg
        self.T = [0] * self.len_T
        self.E = [0] * self.len_T
        self.F = F
        self.R = R
        self.C = C
        self.L = L
        self.CommCost = CommCost
        self.V = V



    def slot_optimize(self, t):
        scheduling_decision = []
        for n in range(self.num_node):
            scheduling_decision.append([0] * self.num_node)
        F_ = self.F
        non_zero_indices = [i_ for i_, x in enumerate(F_) if x != 0]

        xlim = [0, 1, 2]
        popsize = len(non_zero_indices)
        chromlength = self.num_node 
        pc = 0.6 
        pm = 0.1 
        G = 50

        pop = np.zeros((popsize, chromlength))
        for index in range(popsize):
            random_list = generate_random_list(len(non_zero_indices), xlim)
            for index_ in range(len(non_zero_indices)):
                pop[index][non_zero_indices[index_]] = random_list[index_]
        scheduling_decisions, obj = self.calobjvalue_GA(pop, t)

        obj_best = [min(obj)]
        scheduling_decision_best = [scheduling_decisions[np.argmin(obj)]]

        for index in range(1, G):
            scheduling_decisions, obj = self.calobjvalue_GA(pop, t)
            newpop = copyx(pop, obj, popsize)
            newpop = crossover(non_zero_indices, newpop, pc, popsize)
            newpop = mutation(non_zero_indices, newpop, pm, popsize)

            _, new_obj = self.calobjvalue_GA(newpop, t)
            index = np.where(new_obj < obj)
            pop[index] = newpop[index]

            scheduling_decisions, obj = self.calobjvalue_GA(pop, t)
            best_val = min(obj)
            best_scheduling_decision = scheduling_decisions[np.argmin(obj)]
            obj_best.append(best_val)
            scheduling_decision_best.append(best_scheduling_decision)

        best_obj = min(obj_best)
        best_scheduling_decision = scheduling_decision_best[np.argmin(obj_best)]

        return best_scheduling_decision, best_obj

    def calobjvalue_GA(self, pop, t):
        obj = np.zeros(pop.shape[0]) 
        scheduling_decisions = [] 
        for index in range(pop.shape[0]):
            source_nodes = [index_ for index_ in range(self.num_node) if pop[index][index_] == 0]
            sink_nodes = [index_ for index_ in range(self.num_node) if pop[index][index_] == 1]

            non_zero_indices = []
            for index_ in range(self.num_node):
                non_zero_indices.append([])
            for index_ in range(self.num_node):
                if index_ in source_nodes:  
                    if self.F[index_] == 0:
                        non_zero_indices[index_].extend(sink_nodes)
                    else:
                        non_zero_indices[index_].append(index_)  
                        non_zero_indices[index_].extend(sink_nodes)
                elif index_ in sink_nodes: 
                    non_zero_indices[index_].append(index_)
                else:  
                    non_zero_indices[index_].append(index_)

            dim_opt = [0] * self.num_node
            for index_ in range(self.num_node):
                dim_opt[index_] = len(non_zero_indices[index_]) - 1


            if any(x >= 1 for x in dim_opt):
                lb = [0] * sum(dim_opt)
                ub = [0] * sum(dim_opt)
                for index_ in range(self.num_node):
                    for index__ in range(dim_opt[index_]):
                        ub[sum(dim_opt[:index_]) + index__] = self.arrived_lists[t][index_]
                hms = 30
                iter_ = max(ub)
                hmcr = 0.8
                par = 0.1
                bw = 0.1
                nnew = len(lb)
                scheduling_decision, obj[index] = self.Harmony_Search(non_zero_indices, dim_opt, hms, iter_, hmcr, par,
                                                                      bw, nnew, lb, ub, t)
                scheduling_decisions.append(scheduling_decision)
            else:
                obj[index] = 1e10
                scheduling_decision = []
                for _ in range(self.num_node):
                    scheduling_decision.append([0] * self.num_node)
                F_ = np.array(self.F)
                non_zero_indices_ = np.nonzero(F_)[0].tolist()
                non_zero_count = len(non_zero_indices_)
                for n in range(self.num_node):
                    distribution = distribute_evenly(self.arrived_lists[t][n], non_zero_count)
                    for index_ in range(non_zero_count):
                        scheduling_decision[n][non_zero_indices_[index_]] = distribution[index_]
                scheduling_decisions.append(scheduling_decision)

        return scheduling_decisions, obj

    def Harmony_Search(self, non_zero_indices, dim_opt, hms, iter_, hmcr, par, bw, nnew, lb, ub, t):
        pos = []
        score = []
        dim = len(lb)
        for _ in range(hms):
            temp_pos = [random.randint(lb[j], ub[j]) for j in range(dim)]
            pos.append(temp_pos)
            score.append(self.calobjvalue_HS(temp_pos, non_zero_indices, dim_opt, t))

        gbest = min(score)
        gbest_pos = pos[score.index(gbest)].copy()

        for _ in range(iter_):
            new_pos = []
            new_score = []

            for _ in range(nnew):
                temp_pos = []
                for j in range(dim):
                    if random.random() < hmcr:
                        ind = random.randint(0, hms - 1)
                        temp_pos.append(pos[ind][j])
                        if random.random() < par:
                            temp_pos[j] += math.floor(random.normalvariate(0, 1) * bw * (ub[j] - lb[j]))
                    else:
                        temp_pos.append(random.randint(lb[j], ub[j]))
                temp_pos = boundary_check(temp_pos, lb, ub)
                new_pos.append(temp_pos)
                new_score.append(self.calobjvalue_HS(temp_pos, non_zero_indices, dim_opt, t))

            new_pos.extend(pos)
            new_score.extend(score)
            sorted_score = sorted(new_score)
            pos = []
            score = []
            for index in range(hms):
                score.append(sorted_score[index])
                pos.append(new_pos[new_score.index(sorted_score[index])])

            if score[0] < gbest:
                gbest = score[0]
                gbest_pos = pos[0].copy()

        g = [0] * self.num_node
        for index in range(self.num_node):
            if dim_opt[index] == 0:
                g[index] = self.arrived_lists[t][index]
            else:
                g[index] = self.arrived_lists[t][index] - sum(
                    np.array(gbest_pos[sum(dim_opt[:index]):sum(dim_opt[:index + 1])]))
        if all(x >= 0 for x in g):
            scheduling_decision = []
            for n in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            for j in range(self.num_node):
                if dim_opt[j] == 0:
                    scheduling_decision[j][non_zero_indices[j][0]] = self.arrived_lists[t][j]
                else:
                    for index_ in range(dim_opt[j]):
                        scheduling_decision[j][non_zero_indices[j][index_]] = gbest_pos[sum(dim_opt[:j]) + index_]
                    scheduling_decision[j][non_zero_indices[j][-1]] = g[j]
            return scheduling_decision, gbest
        else:
            scheduling_decision = []
            for _ in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            F_ = np.array(self.F)
            non_zero_indices = np.nonzero(F_)[0].tolist()
            non_zero_count = len(non_zero_indices)
            for n in range(self.num_node):
                distribution = distribute_evenly(self.arrived_lists[t][n], non_zero_count)
                for index in range(non_zero_count):
                    scheduling_decision[n][non_zero_indices[index]] = distribution[index]

            gbest = 1e10
            return scheduling_decision, gbest

    def get_T(self, scheduling_decision, t):
        CompTime = [0] * self.num_node
        NumTask_node = [0] * self.num_node
        for n in range(self.num_node):
            for index in range(self.num_node):
                NumTask_node[n] += scheduling_decision[index][n]
            try:
                if NumTask_node[n] != 0 and self.F[n] == 0:
                    raise CustomInterrupt("在没有部署该服务的节点上分配了该任务的请求")
            except CustomInterrupt as e:
                print(f"捕获到自定义中断: {e}")
                log_file_path = os.path.join(os.getcwd(), "app.log")
                logger = Logger(log_file_path, logging.DEBUG)
                logger.error(f"任务调度决策: {scheduling_decision}, 资源分配情况: {self.F}")

            if NumTask_node[n] != 0:
                CompTime[n] = self.C * NumTask_node[n] / self.F[n]
        CompTimeTotal = sum(np.array(CompTime) * np.array(NumTask_node))

        CommTime = []
        for _ in range(self.num_node):
            CommTime.append([0] * self.num_node)
        for n in range(self.num_node):
            for m in range(self.num_node):
                if n != m:
                    CommTime[n][m] = self.L * scheduling_decision[n][m] / self.R[n][m]
        CommTimeTotal = sum(sum(np.array(CommTime) * np.array(scheduling_decision)))
        self.T[t] = CompTimeTotal + CommTimeTotal
        return CompTimeTotal + CommTimeTotal

    def get_E(self, scheduling_decision, t):
        cost = 0
        for n in range(self.num_node):
            for m in range(self.num_node):
                if n != m:
                    cost += self.CommCost[n][m] * scheduling_decision[n][m]
        self.E[t] = cost
        return cost

    def calobjvalue_HS(self, temp_pos, non_zero_indices, dim_opt, t):
        g = [0] * self.num_node
        for index in range(self.num_node):
            if dim_opt[index] == 0:
                g[index] = self.arrived_lists[t][index]
            else:
                g[index] = self.arrived_lists[t][index] - sum(
                    np.array(temp_pos[sum(dim_opt[:index]):sum(dim_opt[:index + 1])]))
        if all(x >= 0 for x in g):
            scheduling_decision = []
            for n in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            for j in range(self.num_node):
                if dim_opt[j] == 0:
                    scheduling_decision[j][non_zero_indices[j][0]] = self.arrived_lists[t][j]
                else:
                    for index_ in range(dim_opt[j]):
                        scheduling_decision[j][non_zero_indices[j][index_]] = temp_pos[sum(dim_opt[:j]) + index_]
                    scheduling_decision[j][non_zero_indices[j][-1]] = g[j]
            T = self.get_T(scheduling_decision, t)
            E = self.get_E(scheduling_decision, t)
            return self.V * T + self.Q[t] * (E - self.E_avg)
        else:
            return 1e10

    def run(self):
        for t in tqdm(range(self.len_T), desc="Running"):
            scheduling_decision, _ = self.slot_optimize(t)
            self.get_T(scheduling_decision, t)
            self.get_E(scheduling_decision, t)
            if t < self.len_T - 1:
                self.Q[t + 1] = max(self.Q[t] + self.E[t] - self.E_avg, 0)
        T_avg = sum(self.T) / self.len_T
        E_avg_ = sum(self.E) / self.len_T
        print("T_avg:", T_avg, "E_avg:", E_avg_)


if __name__ == '__main__':
    input_file = 'variables.txt'
    variables = parse_variables(input_file)

    num_node = variables['num_node']
    arrived_lists = variables['arrived_lists']
    F = variables['F']
    R = variables['R']
    C = variables['C']
    L = variables['L']
    CommCost = variables['CommCost']
    V = 100000
    E_avg = 20
    len_T = variables['len_T']

    print("num_node=", num_node)
    print("arrived_lists=", arrived_lists)
    print("F=", F)
    print("R=", R)
    print("C=", C)
    print("L=", L)
    print("CommCost=", CommCost)
    print("V=", V)
    print("E_avg=", E_avg)

    model = MyLyapunuov(num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V)
    model.run()
