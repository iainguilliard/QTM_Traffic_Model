# A Non-homogeneous Time Mixed Integer LP Formulation for Traffic Signal Control

Iain Guilliard  National ICT Australia  7 London Circuit  Canberra, ACT, Australia  iain.guilliard@nicta.com.auScott Sanner  Oregon State University  1148 Kelley Engineering Center  Corvallis, OR 97331  scott.sanner@oregonstate.eduFelipe W. Trevizan  National ICT Australia  London Circuit  Canberra, ACT, Australia  felipe.trevizan@nicta.com.auBrian C. Williams  Massachusetts Institute of Technology  Massachusetts Avenue  Cambridge, MA 02139  williams@csail.mit.edu4873 words + 8 figures + 0 table + 27 citations  (Weighted total words: 6873 out of 7000 + 35 references)  August 1, 2015

## Abstract

As urban traffic congestion is on the increase worldwide, it is critical
to maximize capacity and throughput of existing road infrastructure
through optimized traffic signal control. To this end, we build on the
body of work in mixed integer linear programming (MILP) approaches that
attempt to jointly optimize traffic signal control over an entire
traffic network and specifically on improving the scalability of these
methods for large numbers of intersections. Our primary insight in this
work stems from the fact that MILP-based approaches to traffic control
used in a receding horizon control manner (that replan at fixed time
intervals) need to compute high fidelity control policies only for the
early stages of the signal plan; therefore, coarser time steps can be
employed to "see" over a long horizon to preemptively adapt to distant
platoons and other predicted long-term changes in traffic flows. To this
end, we contribute the queue transmission model (QTM) which blends
elements of cell-based and link-based modeling approaches to enable a
non-homogeneous time MILP formulation of traffic signal control. We then
experiment with this novel QTM-based MILP control in a range of traffic
networks and demonstrate that the non-homogeneous MILP formulation
achieves (i) substantially lower delay solutions, (ii) improved
per-vehicle delay distributions, and (iii) more optimal travel times
over a longer horizon in comparison to the homogeneous MILP formulation
with the same number of binary and continuous variables.

## Introduction

As urban traffic congestion is on the increase worldwide with estimated
productivity losses in the hundreds of billions of dollars in the U.S.
alone and immeasurable environmental impact [@bazzan2013intro], it is
critical to maximize capacity and throughput of existing road
infrastructure through optimized traffic signal control. Unfortunately,
many large cities still use some degree of *fixed-time*
control [@el2013multiagent] even if they also use *actuated*
or *adaptive* control methods such as
SCATS [@scats80] or
SCOOT [@scoot81]. However, there is further
opportunity to improve traffic signal control even beyond adaptive
methods through the use of *optimized* controllers (that
incorporate elements of both adaptive and actuated control) as
evidenced in a variety of approaches including mixed
integer (linear) programming [@gartner1974optimization,@gartner2002arterial,@lo1998novel,@he2011pamscod,@lin2004enhanced,@han2012link], heuristic
search [@lo1999dynamic,@he2010heuristic], queuing delay with pressure
control [@varaiya2013max] and linear program
control [@li2014coupled], to scheduling-driven
control [@xie2012schedule,@smith2013surtrac], and reinforcement
learning [@el2013multiagent]. Such optimized controllers hold
the promise of maximizing existing infrastructure capacity by finding
more complex (and potentially closer to optimal) jointly coordinated
intersection policies in comparison to heuristically-adaptive policies
such as SCATS and SCOOT. However, optimized methods are computationally
demanding and often do not guarantee *jointly* optimal solutions over a
large intersection network either because (a) they only consider
coordination of neighboring intersections or arterial routes or (b) they
fail to scale to large intersection networks simply for computational
reasons. We remark that the latter scalability issue is endemic to many
mixed integer programming approaches to optimized signal control.

In this work, we build on the body of work in mixed integer linear
programming (MILP) approaches that attempt to jointly optimize traffic
signal control over an *entire traffic network* (rather than focus on
arterial routes) and specifically on improving the scalability of these
methods for large urban traffic networks. In our investigation of
existing approaches in this vein, namely exemplar methods in the spirit
of [@lo1998novel,@lin2004enhanced] that use a (modified) cell
transmission model (CTM) [@daganzo1994cell,@daganzo1995cell] for their
underlying prediction of traffic flows, we remark that a major drawback
is the CTM-imposed requirement to choose a predetermined *homogeneous*
(and often necessarily small) time step for reasonable modeling
fidelity. This need to model a large number of CTM cells with a small
time step leads to MILPs that are exceedingly large and often
intractable to solve.

Our primary insight in this work stems from the
fact that MILP-based approaches to traffic control used in a receding
horizon control manner (that replan at fixed time intervals) need to
compute high fidelity control policies only for the early stages of the
signal plan; therefore, coarser time steps can be employed to "see" over
a long horizon to preemptively adapt to distant platoons and other
predicted long-term changes in traffic flows. This need for
non-homogeneous control in turn spawns the need for an additional
innovation: we require a traffic flow model that permits non-homogeneous
time steps and properly models the travel time delay between lights. To
this end, we might consider CTM extensions such as the variable cell
length CTM [@xiaojian2010urban], stochastic CTM
[@sumalee2011stochastic,@jabari2012stochastic], CTM extensions for better modeling
freeway-urban interactions [@huang2011traffic] including CTM
hybrids with link-based models [@muralidharan2009freeway],
assymmetric CTMs for better handling flow imbalances in merging
roads [@gomes2006optimal], the situational CTM for better
modeling of boundary conditions [@kim2002online], and the
lagged CTM for improved modeling of the flow density
relation [@lu2011discrete]. However, despite the widespread
varieties of the CTM and usage for a range of
applications [@alecsandru2011assessment], there seems to be no
extension that permits *non-homogeneous* time steps as proposed in our
novel MILP-based control approach.

For this reason, as a major contribution of this work to enable our
non-homogeneous time MILP-based model of joint intersection control, we
contribute the queue transmission model (QTM) that blends elements of
cell-based and link-based modeling approaches as illustrated and
summarized in Figure 1. The QTM offers the following key
benefits:

-   Unlike previous CTM-based joint intersection signal
    optimization [@lo1998novel,@lin2004enhanced], the QTM is intended for
    *non-homogeneous* time steps that can be used for control over large
    horizons.

-   Any length of roadway without merges or diverges can be modeled as a
    single queue leading to compact QTM MILP encodings of large traffic
    networks (i.e., large numbers of cells and their associated MILP
    variables are not required *between* intersections). Further,
    the free flow travel time of a link can be modeled exactly,
    independent of the discritizaiton time step, while CTM requires a
    further increased discretization to approach the same
    resolution.

-   The QTM accurately models fixed travel time delays critical to green
    wave coordination as in 
    [@gartner1974optimization,@@gartner2002arterial,he2011pamscod] through the
    use of a non-first order Markovian update model and further combines
    this with fully joint intersection signal optimization in the spirit
    of [@lo1998novel,@lin2004enhanced,@han2012link].

In the remainder of this paper, we first formalize our novel QTM model
of traffic flow with non-homogeneous time steps and show how to encode
it as a linear program for computing traffic flows. Next we proceed to
allow the traffic signals to become discrete phase variables that are
optimized subject to a delay minimizing objective and standard minimum
and maximum time constraints for cycles and phases; this results in our
final MILP formulation of traffic signal control. We then experiment
with this novel QTM-based MILP control in a range of traffic networks
and demonstrate that the non-homogeneous MILP formulation achieves (i)
substantially lower delay solutions, (ii) improved per-vehicle delay
distributions, and (iii) more optimal travel times over a longer horizon
in comparison to the homogeneous MILP formulation with the same number
of binary and continuous variables.
