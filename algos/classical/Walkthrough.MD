
In this section we tackle the classical algorithms and the first fundements of the Reinforcement-Learning Realm, we try to understand the building point, and the limitation of these methods :

# Value Iteration :

The algorithm tries to find the optimal Value function $V^*$ in an iterative way, why since the objective is to find the policy ? because by definition : if $ \forall  s, V_{\pi_1} > V_{\pi_2}  \Rightarrow \mathbb{E}_{\pi_1}(G) > \mathbb{E}_{\pi_2}(G)$ and hence $\pi_1 > \pi_2$.



Convergence guarantee :
- solving Bellman's equation.
- Finite $&$ small set of actions and states --> discretiation needed for our case.
- known enivronement. 

### Discretization : 
 Our environement is supposed to mimic real world, wich is continuous (there is infinitly possible acceleration vectors that can be given), but as discussed before, these type of algorithms necessit a finite set od statae and actions as well : so a necessary discretization is done : 
- Disceretisize the world : the world would be seen as a (NxN) grid, the grids represent the possible places where the drone can be, like a chess piece inside a (chess-board). in the code N=20, meaning we have 20x20 = 400 possible state
- THe set of actions is dicretized as well, for each acceleration

Problems :
- Compute the value function for all states, necessity of knowing the enviroenemnt, not pratctical in real life, we can't run the drone in the whole sky???
- Simulation poor, not practical, huge discretization needed, wich will make the algo impossible to converge. 

