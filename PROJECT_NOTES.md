# Learning Notes

## Core idea
Federated learning separates **where data lives** from **where a shared model is coordinated**. In this simulation, all clients run on one machine, but the code preserves the logical boundary: a client trains only on its own local subset and the server aggregates parameters.

## Questions to answer while experimenting
1. Why should FedAvg weight clients by local dataset size?
2. What changes when only a fraction of clients participates in each round?
3. Why can non-IID data make a locally good update harmful to the global model?
4. How does increasing local epochs change communication efficiency versus client drift?
5. Why is federated learning not automatically privacy-preserving?

## Important caveat
This is a simulation, not a production distributed system. Real FL additionally needs networking, authentication, privacy/security protections, fault handling, resource-aware scheduling, and more.
