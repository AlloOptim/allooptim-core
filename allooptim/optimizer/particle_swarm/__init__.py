"""Particle Swarm Optimization (PSO) for portfolio allocation.

This package implements Particle Swarm Optimization, a population-based
metaheuristic inspired by the social behavior of bird flocking and fish
schooling. PSO searches for optimal portfolio weights by having particles
move through the solution space influenced by their own best positions
and the swarm's best position.

Key features:
- Population-based optimization
- Velocity and position updates
- Social and cognitive learning components
- Early stopping criteria
- Integration with PySwarms library
"""

import logging

logger = logging.getLogger(__name__)
