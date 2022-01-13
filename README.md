# Brownian Control Problem Simulation (BCPSim)

This project was part of my doctorate study at the Technion, under the supervision of [Prof. Rami Atar](https://rami.net.technion.ac.il/). 

It contains the code in python used to create the simulation for the paper *Optimizing buffer size for the retrial queue: two state space collapse results in heavy traffic*
([link](https://webee.technion.ac.il/Sites/People/atar/ata-lev-2.pdf)). 

The paper studies a single queueing system with a finite buffer. Each customer that arrives to find the buffer full is rejected and returns to the queue (with probability p) after exponential time (with a parameter <img src="https://render.githubusercontent.com/render/math?math=\mu">). It also considers the problem of finding the optimal buffer size in heavy traffic limit, with holding and rejecting costs. This simulation was produced in the absence of an analytical solution for general values of <img src="https://render.githubusercontent.com/render/math?math=\mu">. 

This simulation mimics the behavior of the queue. As a function of <img src="https://render.githubusercontent.com/render/math?math=\mu">, it calculates the asymptotically optimal buffer size and value function when considering different cases for the probability parameter p.  

The obtained graphs are shown below and can be found on page 26 of the article, along with clarification of the results.
![](https://github.com/anatlevari/BCPSim/raw/master/image.png)

