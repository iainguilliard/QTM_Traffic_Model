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

## The Queue Transmission Model (QTM)

A Queue Transmission Model (QTM) is the tuple $(\Qset, \Lset, \vecDT, \MatQIN)$,
where $\Qset$ and $\Lset$ are, respectively, the set of queues and lights;
$\vecDT$ is a vector of size $\Nn$ representing the homogeneous, or non-homogeneous, discretization of the problem
horizon $[0,\TMAX]$ and the duration in seconds of the $n$-th time interval is
denoted as $\DT[n]$;
and $\MatQIN$ is a matrix $|\Qset| \times \TMAX$ in which $\QIN{i}{n}$ represents
the flow of vehicles requesting to enter queue $i$ from the outside of the network
at time $n$.

A **traffic light** $\tl \in \Lset$  is defined as the tuple $(\CTMIN{\tl},
\CTMAX{\tl}, \Pset_\tl, \VecPTMIN{\tl}, \VecPTMAX{\tl})$, where:

-   $\Pset_\tl$ is the set of phases of $\tl$;

-   $\CTMIN{\tl}$ ($\CTMAX{\tl}$) is the minimum (maximum) allowed cycle time for
  $\tl$; and

-   $\VecPTMIN{\tl}$ ($\VecPTMAX{\tl}$) is a vector of size $|\Pset_\tl|$ and
  $\PTMIN{\tl}{k}$ ($\PTMAX{\tl}{k}$) is the minimum (maximum) allowed time for
  phase $k \in \Pset_\tl$. 


A **queue** $i \in \Qset$ represents a segment of road that vehicles
traverse at free flow speed; once traversed, the vehicles are vertically stacked
in a stop line queue.
Formally, a queue $i$ is defined by the tuple $(\QMAX{i}, \QDELAY{i}, \QOUT{i},
\Fvec_i, \Prvec_i, \QPset{i})$ where:

-   $\QMAX{i}$ is the maximum capacity of $i$;

-   $\QDELAY{i}$ is the time required to traverse $i$ and reach the stop line;

-   $\QOUT{i}$ represents the maximum traffic flow from $i$ to the outside of
  the modeled network;

-   $\Fvec_i$ and $\Prvec_i$ are vectors of size $\Qn$ and their $j$-th entry
  (i.e., $\FMAX{i}{j}$ and $\FTURN{i}{j}$) represent the maximum flow from queue $i$
  to $j$ and the turn probability from $i$ to $j$ (where $\sum_{j \in
  \Qset}\FTURN{i}{j} = 1$), respectively; and


-   $\QPset{i}$ is the set of  traffic light phases controlling the outflow
  of queue $i$, where the pair, $\left(\ell,k \right) \in \QPset{i}$, denotes phase $k$
  of light $\ell$.


Differently than the CTM [@daganzo1994cell; @lin2004enhanced], the QTM does not assume
that $\DT[n] = \QDELAY{i}$ for all $n$, that is, the QTM can represent
non-homogeneous time intervals (\cref{subfig:example}).
The only requirement over $\DT[n]$ is that no traffic light maximum phase time is
smaller than any $\DT[n]$ since phase changes occur only between time intervals;
formally, $\DT[n] \le \min_{\tl \in \Lset, k \in \Pset_\tl} \PTMAX{\tl}{k}$ for
all $n \in \{1,\dots,\Nn\}$. $n \in \{1,\dots,N\}$.

###Computing Traffic Flows with QTM
In this section, we present how to compute traffic flows using QTM and
non-homogeneous time intervals $\DT[]$.
We assume for the remainder of this section that a *valid* control plan for
all traffic lights is fixed and given as parameter;
formally, for all $\tl \in \Lset$, $k \in \Pset_\tl$, and interval $n \in
\{1,\dots,N\}$, the binary variable $\p{\tl}{k}$ is known a priori and indicates
if phase $k$ of light $\tl$ is active (i.e., $\p{\tl}{k} = 1$) or not on interval
$n$. Each phase $k \in \Pset_\tl$ can control the flow from more than one 
queue, allowing arbitrary intersection topologies to be modelled, including "all red"
 phases as a switching penalty and modeling lost time from amber lights.


We represent the problem of finding the maximal flow between capacity-constrained 
queues as a Linear Program
(LP) over the following variables defined for all intervals $n \in
\{1,\dots,\Nn\}$ and queues $i$ and $j$:

- $\q{i} \in [0,\QMAX{i}]$: traffic volume waiting in the stop line of queue
  $i$ at the beginning of interval $n$;
- $\inq{i} \in [0,\QIN{i}{n}]$: inflow to the network via queue $i$ during
  interval $n$;
- $\outq{i} \in [0,\QOUT{i}]$: outflow from the network via queue $i$ during
  interval $n$; and
- $\f{i}{j} \in [0,\FMAX{i}{j}]$: flow from queue $i$ into queue $j$ during
  interval $n$.


The maximum traffic flow from queue $i$ to queue $j$ is enforced by
\eqref{c:turnProb} and \eqref{c:maxFlow}.
\eqref{c:turnProb} ensures that only the fraction $\FTURN{i}{j}$ of the total
internal outflow of $i$ goes to $j$, and since each $\f{i}{j}$ appears 
on both sides of \eqref{c:turnProb}, the upstream queue $i$ will block if any 
downstream queue $j$ is full. \eqref{c:maxFlow} forces the flow from
$i$ to $j$ to be zero if all phases controlling $i$ are inactive (i.e.,
$\p{\ell}{k} = 0$ for all $\left(\ell,k\right) \in \QPset{i}$).
If more than one phase $\p{\ell}{k}$ is active, then \eqref{c:maxFlow} is
subsumed by the domain upper bound of $\f{i}{j}$.

$$
\begin{align}
\f{i}{j} &\le \FTURN{i}{j} \sum_{k=1}^{\Qn}  \f{i}{k} \tag{C1}\tagconstrain{c:turnProb}\\
\f{i}{j} &\le \FMAX{i}{j} \sum_{\left(\ell,k \right) \in \QPset{i}} {\p{\ell}{k}}
\tag{C2}\tagconstrain{c:maxFlow}
\end{align}
$$

To simplify the presentation of the remainder of the LP, we define the helper
variables $\qin{i}$\eqref{def:qin}, $\qout{i}$\eqref{def:qout}, and
$\tn[n]$\eqref{def:tn} to represent the volume of traffic to enter and leave
queue $i$ during interval $n$, and the time elapsed since the beginning of the
problem until the end of interval $\DT[n]$, respectively.

$$
\begin{align}
%
\qin{i} &= \DT (\inq{i} + \sum_{j=1}^{\Qn} \f{j}{i}) \tag{C3}\tagconstrain{def:qin} \\
%
\qout{i} &= \DT (\outq{i} +  \sum_{j=1}^{\Qn} \f{i}{j})
\tag{C4}\tagconstrain{def:qout}\\
%
\tn[n] &= \sum_{x=1}^{n} \DT[x] \tag{C5}\tagconstrain{def:tn}
%
\end{align}
$$


In order to account for the misalignment of the different $\DT[]$ and $\QDELAY{i}$,
we need to find the volume of traffic that entered queue $i$ between two
arbitrary points in time $x$ and $y$ ($x \in [0,\TMAX]$, $y \in [0,\TMAX]$, and $x
< y$), i.e., $x$ and $y$ might not coincide with any $t_n$ for $n \in
\{1,\dots,N\}$
This volume of traffic, denoted as $\Vol_i(x,y)$, is obtained by integrating
$\qin{i}$ over $[x,y]$ and is defined in \eqref{eq:vol} where $m$ and $w$ are the
index of the time intervals s.t. $\tn[m] \le x < \tn[m+1]$ and $\tn[w] \le y <
\tn[w+1]$.
Because the QTM dynamics are *piecewise linear*, $\qin{i}$ is a step function
w.r.t. time and this integral reduces to the sum of $\qin{i}$ over the intervals
contained in $[x,y]$ and the appropriate fraction of $\qin[m]{i}$ and $\qin[w]{i}$
representing the misaligned beginning and end of $[x,y]$.

$$
\begin{equation}
\Vol_{i}(x,y) =
  (\tn[m+1] - x) \frac{\qin[m]{i}}{\DT[m]}
  + \left( \sum_{k=m+1}^{w-1} \qin[k]{i} \right)
  + (y - \tn[w]) \frac{\qin[w]{i}}{\DT[w]}
\tag{1}\label{eq:vol}
\end{equation}
$$

Using these helper variables, \eqref{c:qUpdate} represents the flow conservation
principle for queue $i$ where $\Vol_i(\tn[n-1]-\QDELAY{i},\tn[n]-\QDELAY{i})$ is
the volume of vehicles that reached the stop line during $\DT[n]$.
Since $\vecDT$ and $\QDELAY{i}$ for all queues are known a priori, the indexes $m$
and $w$ used by $V_i$ can be pre-computed in order to encode \eqref{eq:vol};
moreover, \eqref{c:qUpdate} represents a non-first order Markovian update
because the update considers the previous $w-m$ time steps.
To ensure that the total volume of traffic traversing $i$ (i.e.,
$\Vol_i(\tn[n] - \QDELAY{i}, \tn[n])$) and waiting at the stop line does not
exceed the capacity of the queue, we apply \eqref{c:10}.
When queue $i$ is full, $\qin[n]{i} =0$ by \eqref{c:10}, 
which forces $\f{j}{i}$ to 0 in \eqref{def:qin} and \eqref{def:qout}.
This in turn allows the queue in $i$ to spill back into the upstream queue $j$.

$$
\begin{align}
\q{i} &= \q[n-1]{i} - \qout[n-1]{i} +
\Vol_i(\tn[n-1]-\QDELAY{i},\tn[n]-\QDELAY{i}) \tag{C6}\label{c:qUpdate}\\
\Vol_i(\tn[n] - \QDELAY{i}, \tn[n]) + \q{i} &\le \QMAX{i} \tag{C7}\label{c:10}
\end{align}
$$

As with MILP formulations of CTM (e.g. [@lin2004enhanced]),
QTM is also susceptible to *withholding traffic*, i.e., the
optimizer might prevent vehicles from moving from $i$ to $j$ even though the
associated traffic phase is active and $j$ is not full, e.g., this may
reserve space for traffic from an alternate approach that allows the MILP
to minimize delay in the long-term even though it leads to unintuitive traffic
flow behavior.
We address this well-known issue through our objective function \eqref{eq:objFunc} by
maximizing the total outflow $\qout{i}$ (i.e., both internal and external outflow)
of $i$ plus the inflow $\inq{i}$ from the outside of the network to $i$.
This quantity is weighted by the remaining time until the end of the problem
horizon $\TMAX$ to force the optimizer to allow as much traffic volume as possible
into the network and move traffic to the outside of the network as soon as
possible.

$$
\max 
 \sum_{n=1}^{\Nn} \sum_{i=1}^{\Qn} (\TMAX - \tn + 1) (\outq{i} + \inq{i})
\tag{O1}\label{eq:objFunc}
$$

The objective \eqref{eq:objFunc} corresponds to minimizing delay in CTM models,
e.g., \eqref{eq:objFunc} is equivalent to the objective function (O3) in
[@lin2004enhanced] for their parameters $\alpha = 1$, $\beta = 1$ for the origin cells, and
$\beta = 0$ for all other cells.
Figure 2 depicts this equivalence using the cumulative number
of vehicles entering and leaving a network as a function of time.
The delay experienced by the vehicles travelling through this network (red curve
in \cref{fig:cumu_delay_plot}) equals the horizontal difference at each point
between the cumulative departure and arrival curves (less the free flow travel
time through the network).
Maximizing $\outq{i}$ weighted by $(\TMAX - \tn +1)$ in \eqref{eq:objFunc} is the
same as forcing the departure curve to be as close as possible to the arrival
curve as early as possible; therefore, the area between arrival and departure is
minimized, which in turn minimizes the delay.

To illustrate the representation tradeoff offered by non-homogeneous time
intervals, we computed flows and queue volumes for a fixed signal control plan
derived for homogeneous $\DT[n] = 1s$ (ground truth) using different
discretizations.
Figure 3(a) shows the approximation of the ground truth using
homogeneous $\DT[] = 2.5$ and $\DT[] = 5.0$, and Figure 3(b) using
non-homogeneous time intervals that linearly increases from 1s to 2.5s, i.e.,
$\DT[n] \approx 0.0956n + 0.9044$ for $n \in \{1,\dots,17\}$.
As Figure 3(a) shows, large time steps can be rough approximations
of the ground truth.
Non-homogeneous discretization (Figure 3(b)) exploit this fact to
provide a good approximation in the initial time steps and progressively
decrease precision for points far in the future.

## Traffic Control with QTM encoded as a MILP
In this section, we remove the assumption that a valid control plan for all
traffic lights is given and extend the LP
(\ref{eq:objFunc},\ref{c:turnProb}--\ref{c:10}) to an Mixed-Integer LP (MILP)
that also computes the optimal control plan.
Formally, for all $\tl \in \Lset$, $k \in \Pset_\tl$, and interval $n \in
\{1,\dots,N\}$, the phase activation parameter $\p{\tl}{k} \in \{0,1\}$ becomes
a free variable to be optimized.
In order to obtain a valid control plan, we enforce that one phase of traffic
light $\tl$ is always active at any interval $n$ \eqref{c:onlyOnePhaseOn}, and 
ensure cyclic phase polices where
phase changes follow a fixed ordered sequence \eqref{c:seqPhases}, i.e., if phase $k$ was
active during interval $n-1$ and has become inactive in interval $n$, then
phase $k+1$ must be active in interval $n$.
\eqref{c:seqPhases} assumes that $k+1$ equals 1 if $k = \Pn{}$.

$$
\begin{align}
%
\sum\limits_{k=1}^{\Pn} \p{\ell}{k} &= 1\tag{C8}\tagconstrain{c:onlyOnePhaseOn}\\
%
\p[n-1]{\ell}{k} &\le \p{\ell}{k} + \p{\ell}{k+1}\tag{C9}\tagconstrain{c:seqPhases}
%
\end{align}
$$

Next, we enforce the minimum and maximum phase durations (i.e.,
$\PTMIN{\ell}{k}$ and $\PTMAX{\ell}{k}$) for each phase $k \in \Pset_\tl$ of
traffic light $\tl$.
To encode these constraints, we use the helper variable $\pd{\ell}{k} \in
[0,\PTMAX{\ell}{k}]$, defined by constraints
(\ref{c:pd:incUB}--\ref{c:pd:reset}), that:
(i) holds the elapsed time since the start of phase $k$ when $\p[n]{\ell}{k}$ is
active (\ref{c:pd:incUB},\ref{c:pd:incLB});
(ii) is constant and holds the duration of the last phase until the next
activation when $\p[n]{\ell}{k}$ is
inactive (\ref{c:pd:inactiveUB},\ref{c:pd:inactiveLB}); and
(iii) is restarted when phase $k$ changes from inactive to
active \eqref{c:pd:reset}.
Notice that (\ref{c:pd:incUB}--\ref{c:pd:reset}) employs the *big-M*
method to turn the cases that should not be active into subsumed constraints
based on the value of $\p[n]{\tl}{k}$
We use $\PTMAX{\ell}{k}$ as our large constant since $\pd[n]{\ell}{k} \le
\PTMAX{\ell}{k}$ and $\DT[n] \le \PTMAX{\ell}{k}$.
Similarly, constraint \eqref{c:minPhase} ensures the minimum phase time of $k$ and is
not enforced while $k$ is still active.
Figures 4(a) to 4(c) present an example of how
(\ref{c:pd:incUB}--\ref{c:minPhase}) work together as a function of the time $n$ 
for $\pd[n]{\ell}{k}$; the domain constraint $0 \le \pd[n]{\ell}{k} \le
\PTMAX{\ell}{k}$ for all $n \in \{1,\dots,\Nn\}$ is omitted for clarity.
$$
\begin{align}
%
\pd{\ell}{k} &\le
  \pd[n-1]{\ell}{k} + \DT[n-1] \p[n-1]{\ell}{k} 
  + \PTMAX{\ell}{k} (1 - \p[n-1]{\ell}{k})\tag{C10}\tagconstrain{c:pd:incUB}\\
%
\pd{\ell}{k} &\ge
  \pd[n-1]{\ell}{k} + \DT[n-1] \p[n-1]{\ell}{k}
  - \PTMAX{\ell}{k} (1 - \p[n-1]{\ell}{k})\tag{C11}\tagconstrain{c:pd:incLB}\\
%
\pd{\ell}{k} &\le \pd[n-1]{\ell}{k} + \PTMAX{\ell}{k} \p[n-1]{\ell}{k}
  \tag{C12}\tagconstrain{c:pd:inactiveUB}\\
%
%% TODO understand this issue:  
%  \fnremark{check the n-1 in \eqref{c:pd:inactiveUB}}\\
%
%
\pd{\ell}{k} &\ge \pd[n-1]{\ell}{k} - \PTMAX{\ell}{k} \p{\ell}{k}
  \tag{C13}\tagconstrain{c:pd:inactiveLB}\\
%
\pd{\ell}{k} &\le \PTMAX{\ell}{k}(1 - \p{\ell}{k} + \p[n-1]{\ell}{k})
  \tag{C14}\tagconstrain{c:pd:reset}\\
%
\pd{\ell}{k} &\ge \PTMIN{\ell}{k}(1 - \p{\ell}{k}) \tag{C15}\tagconstrain{c:minPhase}
\end{align}
$$


Lastly, we constrain the sum of all the phase durations for light $\tl$ to be
within the cycle time limits $\CTMIN{\tl}$ \eqref{c:cycleLB} and
$\CTMAX{\tl}$ \eqref{c:cycleUB}.
In both \eqref{c:cycleLB} and \eqref{c:cycleUB}, we use the duration of phase 1
of $\tl$ from the previous interval $n-1$ instead of the current interval $n$
because \eqref{c:pd:reset} forces $\pd[n]{\tl}{1}$ to be 0 at the beginning of
each cycle;
however, from the previous end of phase 1 until $n-1$, $\pd[n-1]{\tl}{1}$ holds
the correct elapse time of phase 1.
Additionally, \eqref{c:cycleLB} is enforced right after the end of the each
cycle, i.e., when its first phase is changed from inactive to active.
The value \eqref{c:cycleLB} and \eqref{c:cycleUB} over time for a traffic light
$\tl$ is illustrated in Figure 4(d).

$$
\begin{align}
%
\pd[n-1]{\ell}{1} + \sum\limits_{k=2}^{\Pn} \pd{\ell}{k} &\ge \CTMIN{\ell}
(\p{\ell}{1} - \p[n-1]{\ell}{1}) \tag{C16}\tagconstrain{c:cycleLB}\\
%
\pd[n-1]{\ell}{1} + \sum\limits_{k=2}^{\Pn} \pd{\ell}{k} &\le \CTMAX{\ell}
\tag{C17}\tagconstrain{c:cycleUB}
%
\end{align}
$$

The MILP that encodes the problem of finding the optimal traffic control plan in
a QTM network is defined
by (\ref{eq:objFunc}, \ref{c:turnProb}--\ref{c:cycleUB}).






## References