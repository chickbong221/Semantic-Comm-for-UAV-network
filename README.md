We investigate a heterogeneous network (HetNet)
including sub-6GHz base stations (BSs), mmWave BSs, and THz
BSs to support enhanced mobile broadband (eMBB) users and
ultra-reliable low-latency communication (URLLC) users. We
particularly investigate a user-centric network in which the users
locally and dynamically select and switch among BSs over time
to achieve their highest utility. Two types of users have different
Quality of Service (QoS) requirements. Thus, we design two types
of utility functions specifically for the eMBB users and URLLC
users. Then, to model the dynamic selection behavior of the users,
we propose to use a fractional game with the power-law memory.
The fractional game allows the eMBB users and the URLLC
users to incorporate their past strategies into their current
selection, thus improving their utility. Furthermore, we consider
the case that the BSs cooperate with each other, and we model the
network selection of the users as a multi-agent problem. Then,
we propose to use a multi-agent deep reinforcement learning
(MADRL) algorithm that enables the URLLC users and eMBB
users to make their network selection decision online to achieve
their long-term utility. Various simulation results are provided
to demonstrate the scalability and effectiveness of the proposed
approaches. Particularly, compared with the classical game, the
fractional game is able to achieve a higher utility but incurs a
higher network adaptation cost. Moreover, the different types of
URLLC users (in terms of latency and reliability requirements)
and the number of URLLC users in the network significantly
affect the total utility and the network selection strategies of the
eMBB users. Importantly, given the cooperation among the BSs,
the the MADRL outperforms both the classical and fractional
games in terms of total network utility.
