This project applies various metaheuristics to the airport gate assignment problem with time windows formulated by Lim et al in 2005 [1]. The study is conducted as part of the advanced seminar "Scheduling with Metaheuristics" at the chair of Operations Management of the TUM School of Management.

The following algorithms were implemented in Python 3.7.5 with numpy 1.17.4 and numba 0.46.0:
* Tabu Search
* Genetic Algorithm
* Memetic Algorithm
* Bee Colony Optimization

Their performance is compared to the commercial solver Gurobi.

The project contains two folders "experiments" and "metaheuristics". In order to start a new experiment, run the
file "experimental_study". Note that the total runtime of our experiment including small, medium and large size
instances is around 22 hours. Further explainations of the folder structure and files are given in the following.

Folder "metaheuristics":
    This folder contains the implementation of Gurobi, Tabu Search, Bee Colony Optimization, Genetic and Memetic
    Algorithm. The instance generation and parameter setting is realized in the file "problem". Furthermore a file
    "gantt", which enables the creation of a gantt chart for a given solution, is part of this folder.

Folder: "experiments":
    This folder contains the file "experimental_study" which starts a new run of an experiment. The result data of
    our conducted experiment was saved to the file "experimental_results.csv". Besides these files there are multiple
    files for visualizing our result data. Each plot has it own file.

[1] Lim, Andrew, Brian Rodrigues, and Yi Zhu. "Airport gate scheduling with time windows." Artificial Intelligence Review 24.1 (2005): 5-31.



