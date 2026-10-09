# Candidate Patent Claims: DriftGate

---

## INDEPENDENT CLAIM 1 (Method Claim - Broadest Defensible Combination)

**1. A computer-implemented method for communication-efficient federated learning across a plurality of distributed edge computing devices communicating with a central aggregation server, the method comprising:**
  - **(a)** receiving, at each distributed edge computing device during a federated communication round, global model parameters from the central aggregation server;
  - **(b)** evaluating local streaming sensor data on each distributed edge computing device using a local drift detection engine to detect a distribution shift;
  - **(c)** classifying, at the distributed edge computing device, a detected distribution shift into a transient drift state or a persistent drift state based on a sliding-window temporal persistence criterion;
  - **(d)** updating an on-device participation finite state machine (FSM) comprising a plurality of operational states including at least a stable state, a transient drift state, and a persistent drift state, wherein:
      - in response to transitioning to the transient drift state, uplink transmission of model updates from the edge computing device to the central aggregation server is suppressed; and
      - in response to transitioning to the persistent drift state, the edge computing device selectively partitions model parameters and transmits only a subset of parameter layers determined to exhibit gradient divergence exceeding a predetermined layer divergence threshold;
  - **(e)** selecting, by the central aggregation server, a participating cohort of edge computing devices for a subsequent round based on a joint scoring function combining client FSM state, detected drift severity, update reliability, and uplink bandwidth utilization; and
  - **(f)** aggregating, at the central aggregation server, received parameter updates using an episode-memory weighting mechanism that scales aggregation coefficients according to historical multi-episode drift resolution quality metrics maintained for each edge computing device.

---

## INDEPENDENT CLAIM 2 (System Claim)

**2. A distributed federated learning system for non-stationary edge sensor environments, comprising:**
  - **a plurality of edge computing devices**, each comprising at least one processor, a local memory, a wireless network interface, and instructions causing the processor to:
      - monitor local sensor inference error rates using an ensemble drift detector;
      - classify detected drift into transient or persistent modalities using a temporal persistence verification window;
      - govern local parameter transmission using a state machine that suppresses uplink transmission during transient drift and transmits only drift-affected model layers during persistent drift;
  - **a central aggregation server** communicatively coupled to the plurality of edge computing devices via a network, comprising a server processor and server memory configured to:
      - record structured drift episodes in an episode-memory registry, wherein each episode record includes episode start round, peak drift severity, affected model layers, and post-drift accuracy recovery trajectory;
      - compute per-client aggregation weights as a product of local sample count, historical episode resolution quality, and state-machine reliability modifiers; and
      - perform layer-selective global model aggregation by combining received parameter updates for drifted layers while maintaining existing global parameter values for untransmitted invariant layers.

---

## INDEPENDENT CLAIM 3 (Non-Transitory Computer-Readable Medium)

**3. A non-transitory computer-readable storage medium comprising instructions that, when executed by one or more processors of a distributed computing system, cause the processors to execute the steps of:**
  - detecting concept drift in local distributed data streams;
  - categorizing detected drift into transient or persistent categories using a multi-round verification window;
  - transitioning an edge client state machine between stable, transient, persistent, adapting, and eligible states;
  - conditionally suppressing transmission or selectively transmitting layer-divergence-gated parameter subsets to an aggregation server; and
  - aggregating received updates on the server using a multi-episode decay-discounted historical resolution weighting function.

---

## DEPENDENT CLAIMS (Claims 4 - 18)

**4. The method of claim 1, wherein the sliding-window temporal persistence criterion of step (c) comprises:**
  - tracking positive drift alarms across a temporal window of $\tau_{wait}$ rounds;
  - computing an alarm density ratio representing the proportion of rounds within $\tau_{wait}$ exhibiting distribution shift; and
  - designating the shift as persistent if the alarm density ratio exceeds a persistence threshold $\gamma_{persist}$, and designating the shift as transient if local inference error normalizes within $\tau_{wait}$.

**5. The method of claim 1, wherein suppressing uplink transmission in step (d) comprises:**
  - continuing local gradient descent optimization on the edge computing device to adapt local parameters to temporary conditions; and
  - maintaining an uplink transmission volume of zero bytes to the central aggregation server for a predefined suppression window of rounds, thereby preventing transient noise from corrupting global model parameters.

**6. The method of claim 1, wherein selectively transmitting only a subset of parameter layers in step (d) comprises:**
  - calculating layer-wise gradient norms $\|\nabla_{\theta_l}\|$ for each layer $l$ of the local model during local training;
  - comparing each layer-wise gradient norm against a baseline moving average $\|\nabla_{\theta_l}^{base}\|$ established during stable operation to derive a layer divergence ratio; and
  - packaging only model layers whose layer divergence ratio exceeds the predetermined layer divergence threshold into an uplink payload, while omitting stable representation layers.

**7. The method of claim 6, wherein the local model comprises a multi-layer neural network structured into a base feature extraction group and a task-specific classification head group, and wherein persistent drift selectively uploads only the classification head group.**

**8. The method of claim 1, wherein the participation finite state machine further comprises an adapting state and an eligible state, wherein:**
  - an edge computing device transitions from the persistent drift state to the adapting state upon detection of distribution stabilization;
  - during the adapting state, the edge computing device transmits head-only parameter updates while local validation accuracy is monitored; and
  - the edge computing device transitions from the adapting state to the eligible state once local validation accuracy recovers above a predetermined recovery threshold $\alpha_{recov}$.

**9. The method of claim 1, wherein the joint scoring function of step (e) evaluates client priority score $S_k$ as:**
  $$S_k = \Phi_{FSM}(k) \times (1 + \beta \cdot D_k) \times \mathcal{R}_k \times (1 - C_k)$$
  where $\Phi_{FSM}(k)$ is a state weight assigning zero priority to transient clients and elevated priority to persistent clients, $D_k$ is detected drift severity, $\mathcal{R}_k$ is client update reliability, and $C_k$ is the fraction of total parameters scheduled for transmission.

**10. The method of claim 1, wherein client update reliability $\mathcal{R}_k$ is computed as a weighted combination of gradient trajectory coefficient of variation across rounds, non-suppression participation frequency, and validation accuracy stability.**

**11. The method of claim 1, wherein the episode-memory weighting mechanism of step (f) calculates an episode coefficient $\mathcal{M}_{ep}(k)$ for client $k$ as:**
  $$\mathcal{M}_{ep}(k) = \frac{\sum_{i=0}^{N-1} \lambda^i \cdot Q_{k,i} \cdot (1 - \rho_{k,i})}{\sum_{i=0}^{N-1} \lambda^i}$$
  where $\lambda \in (0, 1)$ is an episode recency decay factor, $Q_{k,i}$ is the post-drift accuracy recovery ratio achieved during historical episode $i$, and $\rho_{k,i}$ is a recurrence penalty applied if episode $i$ represented an unmitigated recurring oscillation.

**12. The method of claim 1, wherein layer-selective aggregation at the central aggregation server combines parameters $\theta_{l}$ for layer $l$ according to:**
  $$\theta_{global, l} = \frac{\sum_{k \in \mathcal{U}_l} w_k \cdot \theta_{k, l}}{\sum_{k \in \mathcal{U}_l} w_k}$$
  where $\mathcal{U}_l$ is the subset of participating clients that transmitted parameters for layer $l$, and wherein invariant layers not transmitted by selective clients retain their current global server parameters.

**13. The method of claim 1, further comprising applying top-$k$ gradient magnitude sparsification to transmitted layer parameters.**

**14. The method of claim 1, wherein the local drift detection engine comprises an ensemble combination of Adaptive Windowing (ADWIN), Page-Hinkley cumulative sum testing, and Drift Detection Method (DDM) error rate tracking.**

**15. The method of claim 1, wherein the distributed edge computing devices comprise IoT wireless sensor nodes operating under battery power constraints.**
