$$
\newcommand{\tagconstrain}[1]{\label{#1}}
\newcommand{\xspace}{}
\newcommand{\ensuremath}[1]{#1}
\newcommand{\Fmax}[1][n]{\ensuremath{\mathcal{F_{#1}}}}
\newcommand{\Vol}{\ensuremath{V}}
\newcommand{\Matrix}[1]{\ensuremath{\mathbf{#1}}}
\newcommand{\Vector}[1]{\ensuremath{\vec{#1}}}
\newcommand{\Net}{\ensuremath{\mathcal{N}}\xspace}
\newcommand{\Qset}{\ensuremath{\mathcal{Q}}\xspace}
\newcommand{\QPset}[1]{\ensuremath{\mathcal{Q}_{#1}^{\mathcal{P}}}}
\newcommand{\Lset}{\ensuremath{\mathcal{L}}\xspace}
\newcommand{\Pset}{\ensuremath{\mathcal{P}}\xspace}
\newcommand{\Pvec}{\ensuremath{\Vector{P}}\xspace}
\newcommand{\Fvec}{\ensuremath{\Vector{F}}}
\newcommand{\Prvec}{\ensuremath{\Vector{P}r}}
\newcommand{\fvec}[1]{\ensuremath{\mathbf{f}_{#1}}}
\newcommand{\tl}{\ensuremath{\ell}\xspace}
\newcommand{\q}[2][n]{\ensuremath{q_{#2,#1}}}
\newcommand{\qin}[2][n]{\ensuremath{q_{#2,#1}^{\mathrm{in}}}}
\newcommand{\qstop}[2][n]{\ensuremath{q_{#2,#1}^{\mathrm{stop}}}}
\newcommand{\qout}[2][n]{\ensuremath{q_{#2,#1}^{\mathrm{out}}}}
\newcommand{\f}[3][n]{\ensuremath{f_{#2,#3,#1}}}
\newcommand{\inq}[2][n]{\ensuremath{f^\mathrm{in}_{#2,#1}}}
\newcommand{\outq}[2][n]{\ensuremath{f^\mathrm{out}_{#2,#1}}}
\newcommand{\p}[3][n]{\ensuremath{p_{#2,#3,#1}}}
\newcommand{\pd}[3][n]{\ensuremath{d_{#2,#3,#1}}}
\newcommand{\aph}[2][n]{\ensuremath{\alpha_{#2,#1}}}
\newcommand{\tn}[1][n]{\ensuremath{t_{#1}}}
\newcommand{\DT}[1][n]{\ensuremath{\Delta t_{#1}}}
\newcommand{\vecDT}{\ensuremath{\Delta \Vector{t}}\xspace}
\newcommand{\Nn}{\ensuremath{\mathrm{N}}\xspace}
\newcommand{\Qn}{\ensuremath{|\mathcal{\Qset}|}\xspace}
\newcommand{\Ln}{\ensuremath{|\mathcal{\Lset}|}\xspace}
\newcommand{\Pn}[1][\ell]{\ensuremath{|\mathcal{\Pset}_{#1}|}}
\newcommand{\TMAX}{\ensuremath{\mathrm{T}}\xspace}
\newcommand{\QMAX}[1]{\ensuremath{\mathrm{Q}_{#1}}}
\newcommand{\MatQIN}{\ensuremath{\Matrix{I}}\xspace}
\newcommand{\QIN}[2]{\ensuremath{I_{#1,#2}}}
\newcommand{\QOUT}[1]{\ensuremath{\mathrm{F}_{#1}^{\mathrm{out}}}}
\newcommand{\QDELAY}[1]{\ensuremath{\mathrm{T}_{#1}^{\mathrm{prop}}}}
\newcommand{\FMAX}[2]{\ensuremath{\mathrm{F}_{#1,#2}}}
\newcommand{\FTURN}[2]{\ensuremath{\mathrm{Pr}_{#1,#2}}}
\newcommand{\PTMAX}[3][,]{\ensuremath{\mathrm{\Phi}_{#2#1#3}^{\mathrm{max}}}}
\newcommand{\PTMIN}[3][,]{\ensuremath{\mathrm{\Phi}_{#2#1#3}^{\mathrm{min}}}}
\newcommand{\VecPTMAX}[1]{\ensuremath{\Vector{\mathrm{\Phi}}_{#1}^{\mathrm{max}}}}
\newcommand{\VecPTMIN}[1]{\ensuremath{\Vector{\mathrm{\Phi}}_{#1}^{\mathrm{min}}}}
\newcommand{\CTMAX}[1]{\ensuremath{\mathrm{\Psi}_{#1}^{\mathrm{max}}}}
\newcommand{\CTMIN}[1]{\ensuremath{\mathrm{\Psi}_{#1}^{\mathrm{min}}}}
$$

## Empirical Evaluation
  
In this section we compare the solutions for traffic networks modeled as a QTM
using homogeneous and non-homogeneous time intervals w.r.t. to two evaluation criteria:
the quality of the solution and convergence to the optimal solution vs. the number
of time steps.

Specifically, we compare the quality of solutions based on the total travel time and we also
consider the third quartile and maximum of the observed delay distribution.
The hypotheses we wish to evaluate in this paper are (i) the quality of the non-homogeneous solutions is at least as good as the
homogeneous ones when the number of time intervals $\Nn$ is fixed; and 
(ii) the non-homogeneous approach requires less time intervals (i.e., smaller
$\Nn$) than the homogeneous approach to converge to the optimal solution.
In the remainder of this section, we present the traffic networks considered in
the experiments, our methodology, and the results.

### Networks

We consider three networks of increasing complexity (Figure 5): an
avenue crossed by three side streets; a 2-by-3 grid; and a 3-by-3 grid with a
diagonal avenue.
The queues receiving vehicles from outside of the network are marked in
Figure 5 and we refer to them as input queues.
The maximum queue capacity ($\QMAX{i}$) is 60 vehicles for non-input queues and
infinity for input queues to prevent interruption of the input demand due to
spill back from the stop line. 
The traversal time of each queue $i$ ($\QDELAY{i}$) is set at 9s (a distance of
125m with a free flow speed of 50km/h).
For each street, flows are defined from the head of each queue $i$ into the tail of the next
queue $j$;
there is no turning traffic ($\FTURN{i}{j}=1$), and the maximum flow rate
between queues, $\FMAX{i}{j}$, is set at 5 vehicles/s.
All traffic lights have two phases, north-south and east-west, and lights 2, 4
and 6 of network 3 have the additional northeast-southwest phase to control the
diagonal avenue.
For networks 1 and 2, $\PTMIN{\tl}{k}$ is 1s, $\PTMAX{\tl}{k}$ is 3s, $\CTMIN{\tl}$ is
2s, and $\CTMAX{\tl}$ is 6s, for all traffic light $\tl$ and phase $k$.
For network 3, $\PTMIN{\tl}{k}$ is 1s and $\PTMAX{\tl}{k}$ is 6s for all $\tl$ and
$k$; and $\CTMIN{\tl}$ is 2s and $\CTMAX{\tl}$ is 12s for all lights \tl except for
lights 2, 4 and 6 (i.e., lights also used by the diagonal avenue) in which
$\CTMIN{\tl}$ is 3s and $\CTMAX{\tl}$ is 18s.

### Experimental Methodology

For each network, a constant background level traffic is injected in the network
in the first 55s to allow the solver to settle on a stable policy.
Then a spike in demand is introduced in the queues marked as $\spadesuit$
(figure 5) from time 55s to 70s to trigger a policy change.
From time 70s to 85s, the demand is returned to the background level, and then
reduced to zero for all input queues.
We extend the problem horizon $\TMAX$ until all vehicles have left the network.
By clearing the network, we can easily measure the total travel time for all the
traffic as the area between the cumulative arrival and departure curves measured
at the boundaries of the network.
The background level for the input queues are 1, 4 and 2 vehicles/s for queues
marked as $\diamondsuit$, $\clubsuit$ and  $\spadesuit$ (Figure 5(d)),
respectively; and during the high demand period, the queues $\spadesuit$ receive 4
vehicles/s.

For both homogeneous and non-homogeneous intervals, we use the MILP QTM
formulation in a receding horizon manner: a control plan is
computed for a pre-defined horizon (smaller than $\TMAX$) and only a prefix of
this plan is executed before generating a new control plan. 
Figure 6(a) depicts our receding horizon approach and we refer to the
planning horizon as a major frame and its executable prefix as a minor frame.
Notice that, while the plan for a minor frame is being executed, we can start
computing the solution for the next major frame based on a forecast model.

To perform a fair comparison between the homogeneous and non-homogeneous
discretizations, we fix the size of all minor frames to 10s and force it to be
discretized in homogeneous intervals of 0.25s.
For the homogeneous experiments, $\DT[]$ is kept at 0.25s throughout the major
frame; therefore, given $\Nn$, the major frame size equals $\Nn/4$ seconds for the
homogeneous approach.
For the non-homogeneous experiments, we increase $\DT[]$ linearly from the end of the minor
frame for 10s and then hold it constant to the end of the major frame. 
We use two discretizations as shown in Figure 6(b):
Non-homogeneous $\vecDT_1$ from 0.25 to 0.5, and Non-homogeneous $\vecDT_2$
from 0.25 to 1.0. For a given $\Nn > 40$, the major frame size used by this non-homogeneous approach
is $10.375 + 1.25(\Nn-40)$ seconds for $\vecDT_1$, and 
$10.375 + 0.625(\Nn-40)$ seconds for $\vecDT_2$.
Once we have generated a series of minor frames, we concatenate them into a
single plan and compute the flow through the network using the QTM LP
formulation with a fixed (homogeneous) $\DT[]$ of 0.25s.
We also compare both receding horizon approaches against the optimal solution
obtained by computing a single control plan for the entire control horizon
(i.e., $[0,\TMAX]$) using a fixed $\DT[]$ of 0.25s.

For all our experiments, we used Gurobi TM as the MILP solver with
12 threads on a 3.1GHz AMD Opteron TM 4334 processor with 12
cores.
We limit the MIP gap accuracy to 0.1% and the time cutoff for solving a major
frame to 3000s for the receding horizon approaches and unbounded in order to
determine the optimal minimum travel time solution to which all other solutions are compared.
All our results are averaged over five runs to account for Gurobi's
stochastic strategies.

## Results

Figures 7(a), 7(c) and 7(e) show, for
each network, the increase in the total travel time w.r.t. the optimal solution
as a function of $\Nn$.
As we hypothesized, the non-homogeneous discretizations requires less time
intervals (i.e., smaller $\Nn$) to obtain a solution with the same total travel
time, and $\vecDT_2$ converges before $\vecDT_1$.
This is important because the size of the MILP, including the number of binary
variables, scales linearly with $\Nn$; therefore, the non-homogeneous approach can
scale up better than the homogeneous one (e.g., Figure 7(e)).
Also, for homogeneous and non-homogeneous discretizations, finding the optimal
solution of major frames with large $\Nn$ might require more time than our imposed
3000s time cutoff and, in this case, Gurobi returns a feasible control plan that
is far from optimal.
The effect in the total travel time of these poor solutions can be seen in
Figure 7(e) for $\Nn > 120$.

The distribution of the total delay observed by each vehicle while traversing the
network is shown in Figures 7(b), 7(d) and 7(f).
Each group of box plots represents a different value of $\Nn$: when the
non-homogeneous $\vecDT_2$ first converges; when the homogeneous $\DT[]$ first
converges; and the final solution itself.
In all networks, the quality of the solutions obtained using both of the 
$\vecDT_1$ and $\vecDT_2$ and is better or equal than using homogeneous $\DT[]$ for fixed $\Nn$ in both
the total travel time and *fairness*, i.e., smaller third quartile and
maximum delay.

To further illustrate the differences between homogeneous and non-homogeneous
discretizations, Figure 8 shows the cumulative arrival and departure
curves and the how delay evolves over time for $q_1$ of network 2
(Figure 5(b)).
In Figure 8(a), the comparison is done when non-homogeneous $\vecDT_2$
first converges (i.e., point I in Figure 7(c)) and for this
value of $\Nn$, the major frame size in seconds of the non-homogeneous approach is
19.125s longer than the homogeneous one.
This allows the MILP solver to "see" 19s further in the future when using
non-homogeneous discretization and find a coordinated signal policy along the
avenue to dissipate the extra traffic that arrives at time 55s.
The shorter major frame of the homogeneous discretization does not allow the
solver to adapt this far in advance and its delay observed after 55s is much
larger than the non-homogeneous one.
Once the homogeneous $\DT[]$ has converged (Figure 8(b)), it is also
able to anticipate the increased demand and adapt well in advance and both
approaches generate solutions close to optimum (Figure 8(c)).

## Conclusion

In this paper, we showed how to formulate a novel queue transmission model (QTM)
of traffic flow with non-homogeneous time steps as a linear program.  We
then proceeded to allow the traffic signals to become discrete variables subject
to a delay minimizing optimization objective and standard traffic signal
constraints leading to a final MILP formulation of traffic signal control with
non-homogeneous time steps.  We
experimented with this novel QTM-based MILP control in a range of traffic networks
and demonstrated that the non-homogeneous MILP formulation
achieved (i) substantially lower delay solutions, (ii) improved per-vehicle delay distributions,
and (iii) more optimal travel times over a longer horizon 
in comparison to the homogeneous MILP formulation with the same number of binary
and continuous variables.
Altogether, this work represents a
major step forward in the scalability of MILP-based jointly optimized traffic
signal control via the use of a non-homogeneous time traffic models and thus helps
pave the way for fully optimized joint urban traffic signal controllers as an
improved successor technology to existing signal control methods.


Our future work includes learning the QTM parameters (e.g., turn probabilities
$\FTURN{i}{j}$ and expected incoming flows $\QIN{i}{n}$) from loop detector data,
and evaluating the impact in scalability of different non-homogeneous
discretizations and size of the computer cluster used for computing the control
plans. 

## ACKNOWLEDGMENTThis work is part of the Advanced Data Analytics in Transport programme, and supported by
National ICT Australia (NICTA) and NSW Trade & Investment. NICTA is funded by the AustralianGovernment through the Department of Communications and the Australian Research Councilthrough the ICT Centre of Excellence Program. NICTA’s role is to pursue potentially economically
significant ICT related research for the Australian economy. NSW Trade & Investment is the
business development agency for the State of New South Wales.