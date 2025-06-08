# Implement a markov chain discrete stochastic process.
# Finite memory m, and n discrete states
# i.e., Each "effective" state is an array of m integers between 0 and n-1

import torch

def rand_trans(n,m, device="cpu"):
	"""
	Sample a random transition probability matrix from each
	history of m states to the next sequence of m states.

	Params
	------
	n : int
		Vocabulary size (number of discrete states).
	m : int
		Memory of the stochastic process.

	Returns
	-------
	trans : torch tensor (n^m x n^m)
		Transition probabilities from each history of 
		m states to the next sequence of m states.
	"""
	# Transition probabilities must be non-negative and normalized by row
	unnormalized_trans = torch.rand((n**m,n**m), device=device)
	trans = torch.nn.functional.normalize(unnormalized_trans, p=1.0, dim=1)

	return trans

def states_to_idx(states,n,m):
	"""
	Assign each vector of m states an integer index. 

	Params
	------
	states : torch tensor (m,)
		vector of m states 
	n : int
		Vocabulary size
	m : int
		Memory

	Returns
	-------
	idx : int
		Index associated with that set of states.
	"""
	digits = n**(torch.range(m-1))
	return int((states * digits).sum().item())


def idx_to_states(idx,n,m, device="cpu"):
	"""
	Convert integer index back to the vector of states.

	Params
	------
	idx : int
		Index associated with that set of states.
	n : int
		Vocabulary size
	m : int
		Memory

	Returns
	-------
	states : torch tensor (m,)
		vector of m states 
	"""
	states = []
	for i in range(m):
		states.append(idx % n)
		idx //= n
	return torch.tensor(states,device=device)

def sample_future(t, n, m, init, trans, device="cpu"):
	"""
	Given a vector of the first m initial states, sample t states into
	the future and return the sequence. 

	Params
	------
	t : int
		Amount of time to sample into the future.
	n : int
		Vocabulary size (number of discrete states).
	m : int
		Memory of the stochastic process.
	init : torch tensor (m,)
		Array of the first m states of the process.
	trans : torch tensor (n^m x n^m)
		Transition probabilities from each history of 
		m states to the next sequence of m states.

	Returns
	-------
	future : torch tensor (t,)
	"""
	# number of steps to simulate
	steps = int(t // m + 1)

	# initial state distribution
	init_dist = torch.zeros(n**m, device=device)
	init_dist[states_to_idx(init,n,m)] = 1.0

	# Compute each subsequent state distribution up to steps.
	# sample a sequence from these distributions
	samples = []
	dist = init_dist
	for i in range(steps):
		dist = torch.matmul(trans, dist)
		idx = int(dist.multinomial(num_samples=1, replacement=True).item())
		samples.append(idx)

	# Compute sequence of individual states
	states = []
	for idx in samples:
		states.append(idx_to_states(idx,n,m,device=device))

	# return future sequence
	return torch.cat(states, dim=0)




