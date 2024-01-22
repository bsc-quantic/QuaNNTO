# QUAntum Neural NeTworks Optics

QUANNTO is a simulation technique for continuous-variables quantum neural networks (CV QNN) based on the quadratures covariance matrix of a photonic(?)(more general?) quantum system. 

The main idea is to simulate a quantum optics system of N modes or degrees of freedom that is transformed by tunable linear (Gaussian) and non-linear (non-Gaussian) transformations mimicking artificial neural network components like the weight matrix and the activation function. For the outcome, an observable needs to be defined whose expectation value will imitate the prediction of the QNN for a given input.

The key point for the simulation is to use the covariance matrix of the quantum system to keep track of its state evolution during the transformations and be able to obtain the observable expectation value (NN outcome) through the covariance matrix of the final state.

It is a hardware-inspired method that can be reproduced in real optical quantum computers using devices like squeezers, phase shifters, beam splitters and photon additions and subtractions. This is due to the fact that all simulated components of the QNN obey all needed constraints to allow the physical implementation of the model.

All crucial definitions will be given throughout this documentation. However, deeper and more extended explanations can be found in (master's thesis)[] and (cv-quantum info book)[].

## 1. Components

First, let's define the components that build up an artificial neural network.

Given an input $\vec{x}$, one layer of an ANN performs the following transformation over it:

$\tilde{f} (\vec{x}) = \Phi (W \vec{x} + \vec {b})$, 

where $W$ is the weight matrix that linearly transforms the input vector $\vec{x}$, $\vec{b}$ is a real vector related to the bias and $\Phi$ is the activation function allowing the learning of non-linear patterns.

Next, let's map all these components into quantum optics components in order to define the QNN structure.

It is important to point out how the quantum optics operators act over the covariance matrix describing the quantum system in order to be able to simulate the system's evolution using its covariance matrix.

### 1.1. Information representation

The element used to represent the NN information is the quadratures covariance matrix $\sigma$ or $V$ of the $N$-mode quantum system, which is of the form (?)(include V form)

### 1.2. Inputs and neurons

The number of inputs and neurons are directly related in this NN model. In the quantum optics system, they are implemented within its degrees of freedom, this is, the number of modes $N$ of the system.

The NN inputs are encoded into the modes and the neurons' functionalities will affect these modes in order to change the value of the inputs and obtain the desired outputs in the training process.

### 1.3. Weight matrix as Gaussian transformations

The weight matrix is in charge of the linear transformation of the inputs and, thus, it is one of the trainable or tunable components of the NN.

The equivalent component for the weight matrix will be a Gaussian transformation $\hat{G}$ that is made of two passive-optics operators and one squeezing operator built like

$\hat{G} = \hat{\mathcal{U}_2} \hat{\mathcal{S}} \hat{\mathcal{U}_1}$,

where the passive-optics operators $\hat{\mathcal{U}}$ map the covariance matrix of the system by acting congruently on it with its symplectic-orthogonal representation $Q$:

$\hat{\mathcal{U}}: \sigma \rightarrow Q \sigma Q^T$  where $Q = \begin{pmatrix}
X & Y \\
-Y & X
\end{pmatrix}$ with $\begin{cases} XY^T - YX^T = \mathbb{0}_N \\
XX^T + YY^T = \mathbb{1}_N \end{cases}$ ,

as well as the squeezing operator $\hat{\mathcal{S}}$ does with its diagonal representation $Z$:

$\mathcal{\hat{S}}: \sigma \rightarrow Z \sigma Z^T$ with $Z=diag(z_1,...,z_N, z_1^{-1}, ..., z_N^{-1})$.

As any symplectic matrix $S$ can be decomposed as $S = Q_2 Z Q_1$, this results into the fact that the Gaussian operator $\hat G$ transforms the covariance matrix of the quantum system it acts onto with it symplectic representation:

$\hat{G}: \sigma \rightarrow S\sigma S^T \equiv \sigma \rightarrow Q_2ZQ_1 \ \sigma \ Q_1^T Z Q_2^T$.

In conclusion, this last transformation of the covariance matrix of the quantum system would represent the weight matrix of an ANN acting over the inputs.

### 1.4. Bias vector as Gaussian transformation

The bias vector $\vec{b}$ is introduced to generalize any possible linear transformation of the inputs $\vec{x}$. 

Its homologous quantum optics component would be the displacement operator $\hat{\mathcal{D}}$ which, for the QNN simulation, can be neglected because it does not modify the covariance matrix at all:

$\hat{\mathcal{D}}: \sigma \rightarrow \sigma$.

Therefore, the displacement is NOT implemented in the simulator.

### 1.5. Activation function as non-Gaussian transformations

The activation function $\Phi$ is usually a non-linear function allowing the NN model to learn patterns of more complex problems.

The quantum optics equivalence will be photon additions $\hat{a}_k^\dag$ or photon subtractions $\hat{a}_k$ of mode $k$, which are also called ladder operators.

These operators are non-Gaussian transformations of the quantum state and they cannot be simulated classically. 

However, applied to a Gaussian state $\rho_G$, there is a way to compute expectation values made of ladder operators acting over a Gaussian state using its covariance matrix. This method is shown in the next section where output obtaining is explained.

### 1.6. Output calculation with non-Gaussian observable (?)(Check title)

As aforementioned, the output of the QNN will be given by the expectation value of some quantum observable. 

Given a Gaussian state $\rho_G$ where ladder operators act onto, the expectation value of the final non-Gaussian state can be computed as

$Tr[\hat{a}_i^\dag...\hat{a}_j^\dag \hat{a}_j...\hat{a}_i\rho_G] = \frac{1}{K}\sum\limits_{\mathcal{P}}\prod\limits_{\{p_1,p_2\}\in \mathcal{P}} Tr[\hat{a}^\#_{p_1}\hat{a}^\#_{p_2}\rho_G]$, 

where $K$ is the normalization factor due to the non-linear transformations of the ladder operators over $\rho_G$, $\mathcal{P}$ is the set of all perfect matchings of the ladder operators in the trace expression and $\hat{a}^\#_k$ refers to photon addition and/or subtraction in the $k$-th mode.

Using the covariance matrix of $\rho_G$, one can compute the trace of every combination of ladder operators pairs with the following identities where $V_{i,j}$ is the covariance matrix element of the $i$-th row and $j$-th column.

$I_1 = Tr[\hat{a}_j\hat{a}_k \rho_G] = \frac{1}{4}[V_{j,k} - V_{N+j, N+k} + i(V_{j,N+k} + V_{N+j,k})]$

$I_2 = Tr[\hat{a}_j^\dagger\hat{a}_k \rho_G] = Tr[\hat{a}_j\hat{a}_k^\dagger \rho_G] - \delta_{j,k} = \frac{1}{4}[V_{j,k} + V_{N+j, N+k} + i(V_{j,N+k} + V_{N+j,k}) - 2\delta_{j,k}]$

$I_3 = Tr[\hat{a}_j\hat{a}_k^\dagger \rho_G] = Tr[\hat{a}_j^\dagger\hat{a}_k \rho_G] + \delta_{j,k} Tr[\rho_G]$

$I_4 = Tr[\hat{a}_j^\dagger\hat{a}_k^\dagger \rho_G] = \frac{1}{4}[V_{j,k} - V_{N+j, N+k} - i(V_{j,N+k} + V_{N+j,k})]$

Then, defining an observable made of ladder operators, its expectation value can be computed in this way.

For example and the most common observable to use is the number operator of $k$-th mode:

$\left\langle\hat N_k\right\rangle = a_k^\dagger a_k$.

So, for example, the expectation value of $\hat{N}_1$ of a Gaussian state $\rho_G$ with one photon addition on mode one would be

$Tr[\hat N_1 a_1^\dagger \ \rho_G \ a_1] = Tr[a_1 \hat N_1 a_1^\dagger \ \rho_G] = Tr[a_1 a_1^\dagger a_1  a_1^\dagger \ \rho_G]$

and the perfect-matchings formula would give the result of the expectation value, where the normalization factor $K$ would be given by the non-Gaussian state resulting after the photon addition over the Gaussian state:

$K=Tr[\hat a_1^\dag\ \rho_G \ \hat a_1] = Tr[\hat a_1 \hat a_1^\dag\ \rho_G]$.

### 1.7. Layers

As said before, each layer of an ANN does the following transformation in the inputs $\vec{x}$:

$\tilde{f} (\vec{x}) = \Phi (W \vec{x} + \vec {b})$.

Due to the fact that a non-Gaussian state is not classically simulable, to create a QNN with more than 1 layer, the commutator between Gaussian and non-Gaussian operators should be used in order to drag the all the non-Gaussianity to the end and keep all Gaussian transformations at the beginning. This allows to use the perfect-matching formula to obtain the expectation values.


## 2. Complexity

To calculate the complexity, the only focus will be the non-Gaussianity, this is, the expectation value and normalization factor $K$ calculation because the Gaussian part is classically simulable.

Defining the hyperparameters

$\begin{cases}
N \equiv \text{Number of modes} \\
L \equiv \text{Number of layers} \\
A \equiv \text{Number of ladder operators per layer (non-linear part)} \\
M \equiv \text{Number of observables (modes to be measured)} \\
l_M \equiv \text{Number of ladder operators of the observable*}
\end{cases}$

and recalling the trace expression

$Tr[\hat{a}_i^\dag...\hat{a}_j^\dag \hat{a}_j...\hat{a}_i\rho_G] = \frac{1}{K}\sum\limits_{\mathcal{P}}\prod\limits_{\{p_1,p_2\}\in \mathcal{P}} Tr[\hat{a}^\#_{p_1}\hat{a}^\#_{p_2}\rho_G]$, 

the number of different trace expressions of this form is given by

$M\cdot (2N)^{2(L-1)A}$

and the number of perfect matchings per trace expression is given by 

$(2AL + l_M - 1)!!$

resulting into

$M\cdot (2N)^{2(L-1)A}\cdot (2AL + l_M - 1)!!$

summations, each with 

$AL + \frac{l_M}{2} - 1$

products of ladder-pair traces.

This makes a total number of 

$M\cdot (2N)^{2(L-1)A}\cdot (AL + \frac{l_M}{2} - 1) \cdot(2AL + l_M - 1)!!$

floating-point operations for this part.

Also, the normalization factor $K$ has a similar complexity but getting rid of the observables part, becoming 

$(2N)^{2(L-1)A}\cdot (AL - 1) \cdot(2AL - 1)!!$

floating-point operations.

In conclusion, the overall number of operations for each QNN evaluation or prediction is given by the sum of both parts:

$\bigl[\ M\cdot (2N)^{2(L-1)A}\cdot (AL + \frac{l_M}{2} - 1) \cdot(2AL + l_M - 1)!!\ \bigr] + \bigl[\ (2N)^{2(L-1)A}\cdot (AL - 1) \cdot(2AL - 1)!!\ \bigr]$.


Therefore, this is an untractable task for classical computation as modes, layers and non-linearity increase. However, this calculation is easily manageable by parallel and distributed systems because the computations are floating-point summations and products.



