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

|   |   |
|:-----------------------------------:|:-----------------------------------:|
|![(a)](TRB_paper/plots/map_overlay.pdf) | ![(b)](TRB_paper/plots/pw_queues.pdf)|
|(a)|(b)|

**FIGURE 1 (a) Example of a real traffic network modeled using the
  QTM. (b) A preview of different QTM model parameters as a function
  of *non-homogeneous* discretized time intervals indexed by $n$.
  For each $n$, we show the following parameters: the elapsed time
  $t$, the non-homogeneous time step length $\Delta t$, the cumulative
  duration $d$ of two different light phases for $l_6$, the phase $p$
  of light $l_6$, and the traffic volume of different queues $q$
  linearly interpolated between time points.  There is technically a
  binary $p$ for each phase, but we abuse notation and simply
  show the current active phase: $\mathit{NS}$ for *north-south green* and 
  $\mathit{EW}$ for *east-west green* assuming the top of the map is north.
  Here we see that traffic progresses from $q_1$ to $q_7$ to $q_9$
  according to light phases and traffic propagation delay with non-homogeneous time steps
  only at required changepoints.
  We refer to the QTM model section for
  precise notation and technical definitions.**

![](TRB_paper/plots/cumu_plot_final_6l.pdf)

**FIGURE 2 Cumulative arrival (blue) and departure (green) curves, and the
 delay curve (red) resulting from the horizontal difference between the arrival and departure curves, less the free flow travel time. The arrival curve is fixed by the demand profile, and the departure curve is maximized by the objective
function \eqref{eq:objFunc}, which has the same effect as minimizing the area
under the delay curve.**


![a](TRB_paper/plots/convergence.pdf)
![b](TRB_paper/plots/convergence_vari.pdf)

**Figure 3 Approximations of a queue volume obtained using homogeneous
$\vecDT = \left \{1.0,\ldots,1.0 \right \}$ using: (a) homogeneous $\vecDT = \left \{2.5,\ldots,2.5 \right \}$and $\vecDT = \left \{5.0,\ldots,5.0 \right \}$ and (b) non-homogeneous
$\vecDT = \left \{1.0, 1.05, 1.1, 1.16, \ldots , 2.29, 2.41, 2.5 \right \}$
$\DT[n] \approx 0.0956n + 0.9044$ for $n \in \{1,\dots,17\}$.  Here we see
that (b) achieves accuracy in the near-term that somewhat degrades over
the long-term, where accuracy will be less critical for receding horizon control.**


![a](TRB_paper/plots/phase_plot_fig_1.pdf)
![b](TRB_paper/plots/phase_plot_fig_2.pdf)
![c](TRB_paper/plots/phase_plot_fig_3.pdf)
![d](TRB_paper/plots/phase_plot_fig_4.pdf)

**Figure 4 Visualization of constraints (C10--C17)
for a traffic light $\tl$ as a function of time.
(a--c) present, pairwise, the constraints (C10--C15)
for phase $k$ ($\pd{\ell}{k}$ as the black line) and the activation variable
$\p[n]{\ell}{k}$ in the small plot.
(d) presents the constraints for the cycle time of $\tl$ (C16 and
C17), where T.C.T. is the total cycle time and is the left hand side
of both constraints.
For this example, $\PTMIN{\ell}{k}=1$, $\PTMAX{\ell}{k}=3$, $\CTMIN{\ell}=7$, and
$\CTMAX{\ell}=8$.**

![a](TRB_paper/plots/network_1.pdf)
![b](TRB_paper/plots/network_2.pdf)
![c](TRB_paper/plots/network_3.pdf)
![d](TRB_paper/plots/demand_plot.pdf)

**Figure 5 (a--c) Networks used to evaluate the QTM performance.
(d) Demand profile of the queues marked as $\diamondsuit$,
$\clubsuit$, and $\spadesuit$ for our experiments.**

![a](TRB_paper/plots/non_homogeneous_control.png)
![b](TRB_paper/plots/non_homogeneous_discretizations.pdf)
 
**Figure 6 (a) Receding horizon control. In this example, the problem
  horizon $\TMAX$ is 40s. The major frames for MILP optimization are
  discretized in 12 time intervals ($\Nn = 12$) and they span 15s and
  30s for homogeneous and non-homogeneous discretizations,
  respectively.  The minor frames represent the prefix of the 
  major frame MILP optimization that is executed.  The horizon
  recedes by the minor frame duration after each execution.
  (b) The two non-homogeneous discretizations used in the experiments, shown here with a major frame duration of 40s.
  From the end of the minor frame time, $\Delta t$ is linearly interpolated over 10s, from 0.25 to 0.5 
  for Non-homogeneous $\vecDT_1$, and 0.25 to 1.0 for Non-homogeneous $\vecDT_2$. 
  $\Delta t$ is then held constant to the end of the major frame time.**

![a](TRB_paper/results/network_1_plot.pdf)
![b](TRB_paper/results/network_1_box_plot-NEW.pdf)
![c](TRB_paper/results/network_2_plot.pdf)
![d](TRB_paper/results/network_2_box_plot-NEW.pdf)
![e](TRB_paper/results/network_3_plot.pdf)
![f](TRB_paper/results/network_3_box_plot-NEW.pdf)

**Figure 7 Increase in the total travel time w.r.t. the optimal solution as a
function of $\Nn$ (a,c,e) and distribution of the total delay of
each car for different values of $\Nn$ (b,d,f).
For each row, the Roman numeral on top of the box plots corresponds to points on the 
travel time plot marked with the same numeral.
The mean of the total delay is presented as a red square in the box plots.
Plots in the $i$-th row correspond to the results for the $i$-th network in
Figure 5.  Non-homogeneous (NH) achieves much better solutions
at smaller $\Nn$ than Homogeneous (H).**


![a](TRB_paper/plots/network_2_cum_plot_early.pdf)
![b](TRB_paper/plots/network_2_cum_plot_converg.pdf)
![c](TRB_paper/plots/network_2_cum_plot_final.pdf)

**Figure 8 Cumulative arrival and departure curves and delay for queue 1 in the
2-by-3 network (Figure 5(b)).
The labels on top of each plot match the labels in
Figures 7(c).
(c) presents the same curves for the optimal solution.  Non-homogeneous (NH $\Delta \vec{t}_2$)
provides near-optimal signal plans over a longer time horizon than
Homogeneous (H) when the number of time intervals $\Nn$ is small.**

## References