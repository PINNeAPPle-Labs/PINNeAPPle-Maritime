# Physics AI for Maritime: A Reference Landscape

*Physics-informed and scientific machine learning applied to ship hydrodynamics, offshore structures, ocean/wave modeling, port logistics, underwater vehicles, and marine propulsion.*

> **Status note:** This is a living seed document for an open-source reference repository. Claims are sourced from public papers, company documentation, and organizational websites as of September 2026. Where a claim could not be independently verified (e.g., a startup's exact algorithmic approach, or a research collaboration not explicitly documented), it is flagged as such rather than asserted. Contributions, corrections, and additional citations are welcome.

---

## Table of Contents

1. [Academic Research Landscape](#1-academic-research-landscape)
2. [Model Architectures](#2-model-architectures)
3. [Commercial Products & Startups](#3-commercial-products--startups)
4. [Numerical Methods & Solvers (Baselines and Hybridization Targets)](#4-numerical-methods--solvers-baselines-and-hybridization-targets)
5. [Geometry & Mesh Resources](#5-geometry--mesh-resources)
6. [Datasets & Data Generators](#6-datasets--data-generators)
7. [Industry Standards & Validation Context](#7-industry-standards--validation-context)
8. [Open Challenges & Gaps](#8-open-challenges--gaps)
9. [Roadmap: Building the Reference Hub](#9-roadmap-building-the-reference-hub)
10. [Where this reference meets the PINNeAPPle codebase](#10-where-this-reference-meets-the-pinneapple-codebase)
11. [Contributing](#11-contributing)
12. [Sources](#12-sources)

---

## 1. Academic Research Landscape

Physics-informed and scientific ML for maritime applications is a genuinely cross-disciplinary field, sitting at the intersection of naval architecture, computational fluid dynamics (CFD), offshore engineering, control theory, and deep learning. The landscape is fragmented across a few concentrated hubs rather than one dominant lab.

### Key institutions and centers

- **MIT (Massachusetts Institute of Technology)** — the Department of Mechanical Engineering's marine hydrodynamics group has a long lineage of connecting physics-based reduced models with data. **G.D. Weymouth and Dick K.P. Yue** authored the foundational "Physics-Based Learning Models for Ship Hydrodynamics" (2014), which combines physics-derived intermediate representations (e.g., waterline wave profiles, breaking bow waves) with learned regression corrections rather than pure black-box surrogates — an early and often-cited template for what is now called physics-guided ML in this domain. MIT is also the birthplace of **WAMIT** (Wave Analysis MIT), the standard potential-flow panel code for offshore/floating body hydrodynamics (see [Section 4](#4-numerical-methods--solvers-baselines-and-hybridization-targets)).
- **TU Delft (Delft University of Technology)** — maintains an active research line explicitly titled "Physics-based learning models for ship hydrodynamics" in its research portal, continuing the MIT-originated PBLM tradition and applying it to Delft's own hydrodynamics test infrastructure. TU Delft is also the origin of the **SWAN** spectral wave model (see Section 4), which underlies much subsequent ML-hybrid wave-modeling work.
- **NTNU (Norwegian University of Science and Technology) — Centre for Autonomous Marine Operations and Systems (AMOS)**: a Norwegian Centre of Excellence combining marine hydrodynamics, ocean structures, control theory, and autonomous systems research. AMOS's two core research thrusts are (1) autonomous vehicles and robotic systems and (2) "safer, smarter and greener" ships, structures, and operations — explicitly bridging classical marine hydrodynamics with learning-based autonomy. AMOS is one of the largest concentrations of marine-robotics-meets-hydrodynamics researchers in the world and regularly publishes at OMAE, IFAC, and in *Ocean Engineering*.
- **University of Strathclyde (Glasgow)** — Department of Naval Architecture, Ocean and Marine Engineering (NAOME) offers a dedicated MSc in "Marine Engineering with Specialisation in Autonomous Marine Vehicles," with taught modules in deep learning and data analysis applied to intelligent sensing — reflecting Strathclyde's broader research portfolio in ship design optimization and marine autonomy.
- Other contributing groups (based on authorship patterns in the *Ocean Engineering*/*Applied Ocean Research* literature surveyed below): naval architecture departments in China (Shanghai Jiao Tong University's **naoe-FOAM-SJTU** solver group), South Korea (KRISO — Korea Research Institute of Ships & Ocean Engineering, originator of the KCS benchmark hull), and the U.S. Navy-affiliated David Taylor Model Basin (DTMB) — all primarily CFD-centric groups that increasingly publish ML-hybrid papers.

### Landmark and representative papers

- Weymouth, G.D. & Yue, D.K.P., **"Physics-Based Learning Models for Ship Hydrodynamics"** (arXiv:1401.3816) — the conceptual anchor paper for physics-guided (as opposed to purely data-driven) ML in ship hydrodynamics, demonstrated on waterline wave profiles, head-sea ship motions, and 3D breaking bow waves.
- Panda, J.P., **"Machine Learning for Naval Architecture, Ocean and Marine Engineering"** (arXiv:2109.05574) — a broad review surveying ML applications (neural networks, regression trees, random forests, SVMs) across wave prediction, wind-load estimation, offshore-platform damage detection, and ship resistance, with discussion of datasets, input features, and optimization strategies used across the literature. Useful as an entry-point survey for the field circa 2021.
- **"A Scoping Review on Simulation-based Design Optimization in Marine Engineering: Trends, Best Practices, and Gaps"** (arXiv:2404.18654) — a 2024 scoping review specifically on simulation-driven (including ML-surrogate) design optimization workflows in marine engineering; a good complementary source to Panda's review for more recent (2022–2024) work.
- **"ShipNet: A Geometric Deep Learning Surrogate for Real-Time Ship Hydrodynamics"** (arXiv:2606.15356) — geometric/graph-based deep learning applied directly to hull surface meshes for real-time hydrodynamic prediction, representative of the shift from tabular/parametric surrogates toward mesh-native geometric deep learning.
- **"Predicting Ship Responses in Different Seaways using a Generalizable Force Correcting Machine Learning Method"** (arXiv:2405.08033) — a force-correction (physics + ML residual) approach to seakeeping prediction, structurally similar in spirit to the MIT PBLM lineage.
- **"Physics informed neural networks based identification modelling of ship maneuvering motion"** (*Engineering Applications of Computational Fluid Dynamics*, 2025, via Taylor & Francis) — PINN-based system identification for ship maneuvering (Abkowitz/MMG-type models), with an optimal-excitation design method for the identification experiment itself.
- **"Bridging ocean wave physics and deep learning: Physics-informed neural operators for nonlinear wavefield reconstruction in real-time"** (arXiv:2508.03315) — explicitly combines neural operator learning with nonlinear wave physics constraints for real-time wavefield reconstruction, directly relevant to seakeeping and offshore operability forecasting.
- **"DVL-DeepONet: A Physics-Guided Operator Learning for Resilient Underwater Navigation"** (arXiv:2606.23502) — DeepONet-style operator learning applied to Doppler Velocity Log (DVL)-based underwater dead-reckoning, illustrating operator-learning methods crossing over from PDE surrogate modeling into vehicle state estimation.
- **"Deep physics for deep seas: Estimating underwater vehicle dynamics with PINNs"** (*Ocean Engineering*, 2026) and **"Research on model predictive control of autonomous underwater vehicle based on physics informed neural network modeling"** (*Ocean Engineering*, 2024) — PINN-based identification of AUV hydrodynamic coefficients (added mass, damping) directly from field data, feeding into MPC controllers.

### Venues

- **SNAME** (Society of Naval Architects and Marine Engineers) — flagship U.S. professional venue; Transactions and the SNAME Maritime Convention host applied ML/hydrodynamics work, though SNAME's proceedings are less indexed on arXiv than European/Asian counterparts.
- **OMAE** (ASME International Conference on Offshore Mechanics and Arctic Engineering) — the largest annual venue explicitly spanning offshore structures, ocean engineering, and increasingly ML/digital-twin tracks.
- **Journals**: *Ocean Engineering* (Elsevier) and *Applied Ocean Research* (Elsevier) are the two dominant journals carrying the bulk of the PINN/ML-for-hydrodynamics literature cited throughout this document; *Journal of Marine Science and Engineering* (MDPI, open access) is a fast-growing outlet for the same content, and *Engineering Applications of Computational Fluid Dynamics* (Taylor & Francis) frequently carries PINN/CFD-hybrid ship papers.
- **ITTC** (International Towing Tank Conference) — not a paper venue per se but the standards-setting body whose recommended procedures (Section 7) define how CFD/EFD (experimental fluid dynamics) validation, including ML-augmented methods, should be reported.
- The **Gothenburg Workshop** series (2000, 2010) and subsequent **Tokyo/Lisbon CFD Workshops** are recurring benchmark-and-compare events (not standing venues but periodic community exercises) that produced the standard benchmark hulls used across nearly all ML-hydrodynamics papers (Section 5).

---

## 2. Model Architectures

### 2.1 PINNs and neural operators for resistance and seakeeping

- **Physics-informed neural networks (PINNs)** embed governing equations (Navier–Stokes, potential-flow, or reduced maneuvering-model ODEs) as soft constraints in the loss function alongside data. In the maritime literature this appears at multiple levels of physics fidelity:
  - Full CFD-level PINNs remain rare for 3D turbulent ship flows (computationally very hard), but are common for reduced-order and 2D/quasi-2D problems (e.g., nonlinear ship rolling, PINN-based solutions to nonlinear rolling motion equations).
  - **System-identification PINNs** are more mature: PINNs are used to fit maneuvering models (Abkowitz-type, MMG-type) and hydrodynamic derivatives directly from trajectory/sensor data while respecting the structure of the underlying ODE, as in the 2025 "PINN-based identification of ship maneuvering motion" work.
  - **Shaft power / engine load prediction**: "Physics-guided Neural Network-based Shaft Power Prediction for Vessels" (arXiv:2512.20348) blends physical propulsion relationships with learned corrections from operational sensor data — representative of the broader "physics-guided" (as opposed to strict PINN) pattern common in industry-facing papers.
- **Neural operators** (Fourier Neural Operator/FNO, DeepONet, and variants) are increasingly used to learn the *solution operator* of wave and ocean-circulation PDEs rather than a single solution instance, enabling fast re-inference across new initial/boundary conditions:
  - FNO-based regional ocean models (e.g., Jahanmard et al.'s "Principled Fourier Neural Operators for High-Resolution Regional Ocean Modeling," JGR: Machine Learning and Computation, 2025/2026) show FNOs reproducing realistic propagating waves and meandering currents despite no explicit physics being hard-coded — an emulator rather than a hybrid solver, trained on outputs of physics-based ocean models.
  - "FNOtD" (FNO with temporal Dimension) extends the standard FNO to jointly parameterize spatial and temporal Fourier domains, better capturing dispersive wave dynamics.
  - Physics-informed neural operators for **nonlinear wavefield reconstruction** (arXiv:2508.03315) directly target real-time sea-state estimation for ship/offshore operability — a maritime-specific (rather than general oceanographic) application of operator learning.
  - Multiscale neural PDE surrogates for downscaling ocean currents (arXiv:2507.18067) address the resolution gap between coarse global reanalysis and the fine-scale currents relevant to coastal/port operations.

### 2.2 Surrogate models for hull-form optimization

This is the most mature and industrially adopted category of physics-AI in the maritime space:

- Typical pipeline: parametric hull deformation (Free-Form Deformation, geometric moments, or shape-basis methods) → limited set of high-fidelity RANS CFD runs → surrogate (Kriging/Gaussian Process, MLP, deep belief network, or radial-basis network) trained to predict resistance/wave-making drag → surrogate-in-the-loop optimization (genetic algorithms, NSGA-II/III, Bayesian optimization).
- Documented results on the ITTC benchmark hulls: a deep-belief-network surrogate loop on **KCS, KVLCC1/2, and JBC** achieved 3.40%, 3.95%, and 2.74% resistance reductions respectively, with >44% computational efficiency gain versus a fully simulation-driven optimization loop.
- MLP-based multi-hull surrogates using geometric-moment dimensionality reduction allow simultaneous optimization across several hull forms rather than one at a time.
- Bayesian-hyperparameter-tuned MLP surrogates and "intelligent constraint-aware sample selection" methods (2025) are recent refinements aimed at making surrogate-based optimization more sample-efficient — directly addressing the high cost of each RANS CFD training sample.
- Commercial convergence: in February 2025, **Siemens Digital Industries Software** and **Compute Maritime** announced a partnership integrating Compute Maritime's **NeuralShipper** generative-AI platform with Siemens' **Simcenter STAR-CCM+** CFD software, explicitly targeting accelerated vessel-design iteration — a concrete industry signal that surrogate/generative hull design is moving from academic papers into commercial CFD toolchains.

### 2.3 ML for wave-field prediction and forecasting

- Deep learning models (CNN/LSTM/attention-based) trained against **ERA5** reanalysis and **NDBC buoy** ground truth for significant wave height (Hs) forecasting; comparative studies report ML models outperforming conventional (physics-based spectral) forecasts at roughly 24 of ~34 evaluated stations in one 2024 study, with remaining stations still favoring traditional models — indicating ML has not uniformly displaced physics-based wave forecasting but is competitive in many regimes.
- "SCAWaveNet" (spatial-channel attention network, arXiv:2507.00701) targets global significant-wave-height retrieval directly from satellite/altimetry-style inputs, reporting RMSE ≈ 0.432 m and correlation ≈ 0.754 against buoy validation.
- "Data-Driven Reconstruction of Significant Wave Heights from Sparse Observations" (arXiv:2509.19384) addresses the practically important problem of reconstructing full wave fields from sparse in-situ sensors — directly relevant to offshore platform and port operability monitoring where dense buoy coverage is unavailable.

### 2.4 Graph neural networks for mooring / offshore structural analysis

- The literature here is still dominated by conventional (fully-connected) **artificial and deep neural networks** rather than graph neural networks specifically — multiple papers on mooring-line fatigue (floating offshore wind turbines, TLPs, FPSOs) use feedforward ANNs/DNNs and sequence-to-sequence deep learning trained on simulated or indirect (platform-motion) response data to predict fatigue damage without needing costly underwater sensors. Reported DNN fatigue-detection accuracy on tension-leg-platform mooring: 94.6–99.3% against simulation ground truth.
- A dedicated **GNN** literature specifically for mooring/offshore topology (exploiting the graph structure of multi-line mooring systems or jacket/truss structures) does not yet appear to be well-established in the public record as of this research pass — this is flagged as an **open gap** rather than an established sub-field (see Section 8). GNNs are more established in adjacent offshore-wind structural health monitoring and in the general structural-engineering ML literature, and are a natural but still under-explored extension for mooring networks and jacket structures.
- Adjacent, more mature work: "Real-Time Digital Twin for Structural Health Monitoring of Floating Offshore Wind Turbines" (*J. Marine Science and Engineering*, 2025) and reduced-order hydro-elastic digital twins (validated on a 5 MW semi-submersible, <1% relative error on FEM displacement/stress, ~100x faster than full FEM) show where offshore structural ML is headed even without explicit graph architectures yet.

### 2.5 Digital twins for vessel performance monitoring

Covered in depth under commercial products (Section 3) since this category is now predominantly industry-driven (Kongsberg Kognitwin, DNV, Wärtsilä) rather than academic-paper-driven. The general architecture pattern across vendors: physics-based hull/propulsion/engine models + streaming sensor data + ML-based drift/anomaly detection + predictive maintenance and performance-degradation forecasting (fouling, engine wear).

### 2.6 ML for propeller design

- Surrogate-driven optimization is well established: Kriging/neural-network surrogates for open-water propeller performance (thrust/torque coefficients, efficiency) coupled to NSGA-II/III or other evolutionary optimizers; one comprehensive feature-based ML surrogate reports ~1000x speedup over direct CFD evaluation.
- A 2025/2026 arXiv paper ("AI-Driven Performance-to-Design Generation and Optimization of Marine Propellers") inverts the typical forward (geometry→performance) surrogate into a **direct performance-to-design generative pipeline**, built on a physics-based data-generation pipeline producing >20,000 four- and five-bladed propeller geometries — a generative-design pattern mirroring what's happening in aerodynamic shape design more broadly.
- Human-in-the-loop ML: one method couples interactive designer judgment on cavitation with an ML model trained to predict/replicate that cavitation assessment, feeding it back into the optimization loop — a hybrid expert-knowledge + ML approach distinct from pure black-box optimization.
- A unified cross-series propeller design method using ML (2024, *Ocean Engineering*) targets generalizing across different propeller series (e.g., Wageningen B-series and others) rather than requiring a bespoke surrogate per series.

### 2.7 Reinforcement learning for autonomous vessel navigation with physics constraints

- Deep RL (DQN, A3C, PPO, actor-critic variants) is the dominant paradigm for **COLREGs-compliant collision avoidance**, typically combining:
  - A **ship-domain / physics-based safety model** (hexagonal or ellipse-based ship domains, kinematic/maneuvering models respecting real turning radii and stopping distances) to define collision risk, encoded either in the reward function or as a hard constraint layer.
  - RL policies (PPO being the most recent favored algorithm) that must simultaneously satisfy COLREGs give-way/stand-on rules, minimize path deviation, and remain dynamically feasible given the vessel's actual maneuverability — this "rule + physics + learning" combination is the maritime analogue of trajectory-constrained RL in autonomous driving.
  - Multi-ship / multi-agent RL with inter-agent communication for cooperative collision avoidance (arXiv:2410.21290) extends single-ship COLREGs compliance to fleet-level coordination.
  - "Trust, Geometry, and Rules" (arXiv:2605.26974) frames USV navigation safety explicitly around credibility-aware RL under uncertainty — pointing toward growing attention to robustness/verifiability, not just nominal-case performance.
- For underwater vehicles, RL (e.g., Soft Actor-Critic for simultaneous AUV control and guidance) is combined increasingly with **physics-informed world models**: "Physics-informed compact world models for predictive control in AUV terminal docking under communication constraints" embeds simplified hydrodynamic priors directly into the learned world model to improve sample efficiency for a genuinely hard, sparse-data regime (docking under comms blackout).

---

## 3. Commercial Products & Startups

| Company | What it does (maritime physics-AI relevant) |
|---|---|
| **DNV** | The world's largest maritime classification society. Runs active digital-transformation and AI initiatives: digital twins for hull condition monitoring (virtual model built at design phase, updated with in-service data to refine class-relevant structural/condition assessments); **RuleAgent**, an AI tool (LLM-based) to help crews/managers query DNV class rules faster; broader "AI-ready organization" guidance for the industry. DNV's rules and joint industry projects are also a primary reference point for how simulation-based/AI design tools get recognized for class approval (see Section 7). |
| **Kongsberg Digital — Kognitwin** | Digital twin platform (renamed from earlier Kongsberg Digital Twin branding to **Kognitwin**) combining physics-based simulation models, cloud APIs, hybrid analytics/ML using synthetic data, and what-if scenario simulation. The maritime variant merges vessel sensor data with external context (weather, traffic) into one operational picture for shipowners/operators; supports condition-based maintenance and performance prediction. Originally built for energy-sector assets and extended to maritime. |
| **Wärtsilä** | Marine engine/propulsion OEM with its own digital/performance-optimization offerings (e.g., voyage and fleet optimization solutions) built around physics-based propulsion and hull models combined with operational data analytics; a major propulsion-system data source for hybrid physics+ML performance models industry-wide. |
| **ABB Marine — ABB Ability™ Marine Pilot** | Suite of digital bridge/autonomy products: **Marine Pilot Vision** (sensor fusion of radar, satellite, LIDAR, gyro into real-time situational-awareness visualization, including a "virtual model of the ship superimposed on real surroundings"); **Marine Pilot Control**, used in a 2018 remote-piloting trial of the ice-class ferry *Suomenlinna II* in Helsinki harbor — an early landmark demonstration of remote/autonomous vessel control. |
| **NAPA** | Finland-based ship design and operations software company; NAPA's core naval-architecture calculation package is a long-standing industry-standard tool (stability, hull design), increasingly paired with data-driven services (route/performance optimization) and partnerships (e.g., NAPA–CADMATIC integration for design workflows). |
| **Windward** | Maritime risk-intelligence/analytics company. Fuses AIS with satellite SAR/EO/RF detections and behavioral-analytics models (reportedly 15+ proprietary AI models) to detect "dark" vessels (AIS switched off/manipulated), sanctions evasion, IUU fishing, and abnormal port/ship-to-ship behavior — an AIS-and-remote-sensing analytics company rather than a hydrodynamics/physics-simulation company, but a major consumer/producer of AIS-derived datasets relevant to Section 6. |
| **Orca AI** | Israeli startup (founded 2018 by Yarden Gross and Dor Raviv) providing AI/computer-vision-based situational awareness and collision-avoidance decision support; deployed on 1,200+ vessels (customers reportedly include MSC, NYK, Scorpio, Seaspan); partnered with The Nippon Foundation/DFFAS on what is described as the world's first commercial autonomous voyage (2022); raised a $72.5M Series B round. Primarily perception/vision-AI rather than physics-simulation, but directly feeds into physics-constrained collision-avoidance systems (Section 2.7). |
| **Sea Machines Robotics** | U.S. company providing autonomous-command and remote-command systems (perception + control stack) for commercial and workboat vessels; frequently named alongside Orca AI as a leading autonomous-navigation vendor. |
| **Nautilus Labs** | Voyage/fleet optimization platform (offices in New York, Singapore, Paris, London). Builds per-vessel ML models tuned to individual hull geometry, propeller configuration, and historical performance, combined with weather routing data (via ChartWorld's route database) to produce daily speed/route recommendations. Reports average fuel savings of 10–12% per voyage (up to 30% in some cases), aimed at CII (Carbon Intensity Indicator) compliance. |
| **ShipIn Systems** | Maritime AI-camera/CCTV analytics company; **FleetVision** platform (Bridge Module) analyzes bridge video feeds for watchkeeping compliance, fatigue/distraction, and unsafe-practice detection, integrating with navigation/weather/machinery sensor data. CEO-reported figures: 42% incident reduction, 17% increase in bridge-manning compliance in deployments — vision/behavioral AI for safety rather than hydrodynamics, included here as a maritime-AI-adjacent safety product. |
| **Sofar Ocean** | Operates the largest privately-owned network of marine weather sensors (Spotter buoys), feeding proprietary and third-party marine weather/wave forecast products; a significant real-world ocean/wave *data* provider (complementing NOAA/ERA5) rather than a modeling company per se. |
| **Ansys** (Fluent, Mechanical) and **Siemens** (Simcenter STAR-CCM+) | The two dominant commercial CFD/FEA vendors serving marine/offshore engineering — used for detailed resistance, seakeeping, slamming, green-water, and offshore wave-loading simulation. Siemens' STAR-CCM+ is notable for the 2025 generative-AI integration with Compute Maritime's NeuralShipper (Section 2.2), a concrete commercial example of physics-AI entering mainstream marine CFD tooling. |
| **FORCE Technology** | Danish maritime technology/testing institute providing towing-tank testing, CFD, and simulation services to the shipping/offshore industry; a validation and applied-R&D partner rather than a software product vendor per se. |
| **Compute Maritime — NeuralShipper** | Generative-AI platform for vessel design acceleration, integrated into Siemens Simcenter STAR-CCM+ (Feb 2025 partnership) — one of the more concrete "generative design meets marine CFD" commercial products identified in this research. |

**Not independently verified in this pass:** "Prospection" and "Deep Vision" (offshore) as named in the original brief could not be confirmed as identifiable companies through this research; they are flagged rather than described, and should be verified (or removed) before publication in the repo. If the intended companies have different canonical names, contributors should correct this table.

---

## 4. Numerical Methods & Solvers (Baselines and Hybridization Targets)

Physics-AI in maritime almost always sits on top of, or is validated against, one of these established numerical tools. They are the natural "ground truth generators" and baseline comparators for any ML/PINN work in this space.

- **Potential-flow / boundary element (panel) methods**
  - **WAMIT** (Wave Analysis MIT) — developed at MIT, first released 1987; the de facto standard linear/second-order potential-flow BEM code for wave loads and dynamic response of floating/fixed offshore structures and ship seakeeping. Solves boundary integral equations via panel discretization of the wetted hull/structure surface.
  - Other BEM/panel codes exist in the same family (e.g., academic and in-house codes at various institutes) but WAMIT is the most widely cited commercial reference.
- **RANS CFD for ship hydrodynamics**
  - **OpenFOAM** — the dominant open-source CFD platform; extended for naval hydrodynamics via community packages including the **Naval Hydro Pack** (foam-extend based; wave/current loading on ships and offshore structures, coastal wave dynamics), **naoe-FOAM-SJTU** (Shanghai Jiao Tong University; wave generation/absorption, 6-DOF motion, mooring, overset grids, FSI), and **NavyFOAM** (U.S. Navy-oriented, high parallel-efficiency variant for wave-body interaction).
  - **STAR-CCM+** (Siemens Simcenter) — leading commercial RANS/CFD code for marine and offshore use, now with generative-AI integration via NeuralShipper (see above).
  - **SHIPFLOW** (FLOWTECH International, Sweden) — a ship-hydrodynamics-specific CFD/potential-flow hybrid code widely used in industry and academia for resistance and self-propulsion prediction; frequently used as an ML-training-data generator in the surrogate-optimization literature.
  - **ANSYS Fluent** — general-purpose commercial CFD, also used for marine/offshore flow and structural (ANSYS Mechanical) coupling.
- **Spectral wave models**
  - **SWAN** (Simulating WAves Nearshore) — third-generation phase-averaged spectral wave model developed at **TU Delft**; includes shallow-water effects (depth-induced breaking, bottom friction, triad wave-wave interaction); the standard nearshore/coastal wave model.
  - **WAVEWATCH III** — NOAA/NCEP's third-generation spectral wave model, the standard for global/regional deep-water wave forecasting; often nested with SWAN for coastal refinement (SWAN can ingest WAVEWATCH III or WAM boundary spectra).
  - **WAM** — an earlier third-generation wave model, still used in intercomparison studies alongside SWAN/WW3.
- **Mooring / offshore structural analysis**
  - **OrcaFlex** (Orcina) — the industry-standard time-domain dynamic analysis code for mooring systems, risers, umbilicals, and other offshore flexible/rigid multi-body systems under wave/current loading; the typical source of training/labeled data for mooring-fatigue ML models (Section 2.4).
- **Seakeeping codes**: typically strip-theory or 3D panel-based seakeeping solvers (often built on the same BEM foundations as WAMIT) used to generate response-amplitude-operator (RAO) training data for ML seakeeping surrogates.

**Pattern across the literature**: essentially every ML/PINN paper surveyed either (a) trains against a *subset* of full RANS CFD runs (from OpenFOAM, STAR-CCM+, or SHIPFLOW) to build a cheaper surrogate, (b) uses potential-flow codes (WAMIT-class) as a lower-fidelity but very cheap physics baseline that ML models learn to correct, or (c) uses spectral wave-model output (SWAN/WW3/ERA5, which itself assimilates model output) as ground truth for wave/ocean ML.

---

## 5. Geometry & Mesh Resources

Standardized benchmark hull geometries are central to reproducibility in this field — nearly every hull-form-optimization or resistance-prediction ML paper surveyed above validates against one or more of these:

- **KCS — KRISO Container Ship**: developed by KRISO (Korea Research Institute of Ships & Ocean Engineering); a modern (bulbous bow and stern, ~1997-era) container-ship hull conceived explicitly to support both flow-physics research and CFD validation. Introduced with full experimental (EFD) data and ITTC-recommended uncertainty analysis at the **Gothenburg 2000 Workshop**. The single most-used benchmark hull in the ML-for-ship-hydrodynamics literature.
- **KVLCC1 / KVLCC2 — KRISO Very Large Crude Carrier**: companion full-form (tanker-type) benchmark hulls from the same Gothenburg 2000 workshop family, used to test full-block-coefficient hull behavior distinct from KCS's slender container-ship form.
- **DTMB 5415**: a David Taylor Model Basin (U.S. Navy) combatant hull form, extensively used for hull-form optimization studies (it has a well-documented sonar dome/transom geometry that makes it a demanding optimization test case) and also introduced with Gothenburg 2000 validation data.
- **JBC — Japan Bulk Carrier**: a more recent (post-Gothenburg 2000, associated with subsequent Tokyo/related CFD workshops) benchmark bulk-carrier hull used for RANS CFD calm-water resistance and self-propulsion validation, now standard alongside KCS in surrogate-optimization benchmarking (see the KCS/KVLCC1/JBC resistance-reduction results cited in Section 2.2).
- **Gothenburg Workshop series** (2000, with a 2010 successor and related Tokyo/Lisbon CFD workshop events) — the recurring community benchmarking exercises that established the above hulls as the shared reference geometries, with experimental data assessed via ITTC uncertainty-analysis procedures. This is the closest maritime-CFD analogue to, e.g., ImageNet in computer vision: a shared, versioned reference dataset the whole field calibrates against.
- **Offshore platform geometry standards**: unlike ship hulls, offshore platform geometries (semi-submersibles, TLPs, jack-ups, spar buoys) are less standardized around a small shared benchmark set; OMAE and API/ISO offshore design standards define load-case and structural-design methodology more than shared public reference geometries. This is a notable **asymmetry and gap** relative to the ship-hydrodynamics side of the field (flagged again in Section 8).

**Practical repo action item**: the KCS, KVLCC2, DTMB 5415, and JBC hull surface geometries (and associated ITTC EFD validation data) are publicly available through NMRI (National Maritime Research Institute, Japan), SIMMAN, and the Gothenburg/Tokyo workshop archives; a maritime physics-AI repo should mirror or clearly link to canonical, license-checked sources for these rather than redistributing scraped copies without attribution.

---

## 6. Datasets & Data Generators

- **ITTC benchmark databases**: ITTC maintains recommended-procedure documents (e.g., **7.5-03-02-01**, **7.5-03-02-02** "Benchmark Database for CFD Validation," **7.5-03-01-01**, **7.5-03-02-03**) that both define validation methodology and list/reference the surface-ship benchmark datasets (largely sourced from the Gothenburg/Tokyo CFD workshops) used for resistance and propulsion CFD validation. The **Specialist Committee on CFD in Marine Hydrodynamics** curates the benchmark experiment list, including required outputs and uncertainty levels.
- **Public towing-tank data**: experimental (EFD) data for KCS, KVLCC1/2, DTMB 5415, and JBC from the Gothenburg 2000 Workshop and successor events, with ITTC-recommended uncertainty quantification applied — the primary "real" (non-CFD) ground truth used to validate both CFD codes and ML surrogates trained on CFD data.
- **Synthetic CFD data generation for hull forms**: the dominant data-generation pattern in the surrogate-optimization literature (Section 2.2) — parametric hull perturbation (FFD, geometric-moment basis) combined with batches of RANS CFD runs (OpenFOAM/STAR-CCM+/SHIPFLOW) to build training sets of a few hundred to a few thousand hull-variant/performance pairs; propeller-design work follows the same pattern (e.g., the >20,000-geometry synthetic propeller dataset referenced in Section 2.6).
- **Ocean/wave reanalysis datasets**:
  - **ERA5** (ECMWF) — the standard global atmospheric/ocean-wave reanalysis dataset, widely used as ML training/reference data for wave-height forecasting; validated against NDBC buoys with reported average correlation ≈0.954 globally, though ERA5 is documented to underestimate significant wave height at the upper (extreme) end of the distribution — an important caveat for any model trained on ERA5 as ground truth for extreme sea states (ties into Section 8's data-scarcity-in-extremes gap).
  - **NOAA NDBC** (National Data Buoy Center) — the standard U.S. in-situ buoy wave/met-ocean observation network, used as the empirical validation layer against both ERA5 and ML wave-forecast models.
  - **Sofar Ocean Spotter buoy network** — a large privately-owned real-time ocean sensor network (Section 3), a growing complementary/commercial data source alongside NOAA/ERA5.
- **AIS (Automatic Identification System) datasets**: AIS provides tracked vessel position/kinematic/identity messages, originally for collision avoidance, now broadly used for traffic analysis, trajectory prediction, and anomaly detection. Notable resources:
  - A published high-quality AIS dataset spanning ~19,016 vessels across six ship types and ~640 million AIS messages (cited in the AISdb-related literature).
  - **AISdb** — an open-source toolkit (SQLite for local use, PostgreSQL for server use) for storing, indexing, and integrating AIS data with environmental datasets, aimed at reproducible ocean-traffic research.
  - **AISHub** — a community data-sharing network aggregating AIS feeds from many independent receivers.
  - A published North Sea vessel-traffic dataset (PMC/NCBI-indexed) is an example of a regional, research-ready AIS dataset.
  - Data-quality caveat consistently flagged in the literature: AIS messages are frequently noisy/incomplete (damaged, duplicated, or inaccurate due to equipment/human error), and can be deliberately manipulated or switched off ("dark" vessels) — the exact problem Windward's analytics platform targets (Section 3).

---

## 7. Industry Standards & Validation Context

- **Classification societies (DNV, ABS, Lloyd's Register)**: these bodies set and enforce the rules under which ships and offshore structures are designed, built, and certified. As of this research pass:
  - DNV has launched **RuleAgent**, an AI (LLM-based) assistant to help crews and shipmanagers navigate class rules more quickly — an operational/compliance AI tool rather than a design-simulation-approval mechanism.
  - Class societies generally permit "synthetic signals or simulated test conditions" under defined circumstances, provided test equipment is maintained/calibrated and appropriately documented — indicating simulation-based evidence is not categorically excluded from certification processes, though specific, detailed public guidance on how ML-based (as opposed to conventional physics-simulation-based) design tools are qualified for class approval was **not found in this research pass** and should be verified directly against current DNV/ABS/LR rule documents (DNV-CG/DNV-RU series, ABS Guides, LR Rules) before the repo asserts specifics.
  - This is a genuine, currently underserved documentation gap: a maritime physics-AI reference repo could add real value by systematically tracking how each major class society's rules treat ML-augmented or simulation-based design and monitoring tools, since this information is scattered across individual rule documents rather than centrally summarized anywhere public.
- **ITTC (International Towing Tank Conference)**: the primary international standards body for towing-tank testing and CFD verification/validation methodology in ship hydrodynamics. Its **Recommended Procedures and Guidelines** (numbered series, e.g., 7.5-03-xx-xx) define uncertainty analysis methods, CFD verification/validation procedures, and the benchmark database structure referenced throughout Sections 5–6. Any physics-AI paper or tool claiming CFD-comparable accuracy should be checked against, and ideally explicitly reports compliance with, relevant ITTC V&V procedures.
- **IMO (International Maritime Organization) — MASS Code**: in May 2026 the IMO adopted the first global **International Code of Safety for Maritime Autonomous Surface Ships (MASS Code)**, effective 1 July 2026. It is presently **voluntary** (non-mandatory) for at least a two-year trial period, providing a goal-based framework for design, certification, and operation of autonomous ships (navigation, connectivity, remote operations, fire safety, search-and-rescue, cybersecurity, and Remote Operation Centres). A mandatory version under SOLAS is targeted for adoption around 2030 and entry into force around 1 January 2032. This is directly relevant to any physics-AI work underpinning autonomous-navigation systems (Section 2.7, Section 3's Orca AI/Sea Machines/ABB entries): those systems will eventually need to demonstrate compliance with MASS Code safety/certification requirements, not just algorithmic performance.
- **Full-scale vs. model-scale validation practice**: ITTC's traditional extrapolation procedure (Froude-scaled model tests + empirical friction-line correction for Reynolds number mismatch) remains the default industry validation path, but full-scale CFD is increasingly used as an alternative that avoids extrapolation entirely — with full-scale CFD reporting prediction errors of roughly −10% to +4% versus towing-tank accuracy of roughly −6% to +11% in one comparative study. This scale-effect question is itself one of the central open challenges for ML surrogates trained predominantly on model-scale data (see Section 8).

---

## 8. Open Challenges & Gaps

1. **Full-scale vs. model-scale extrapolation uncertainty.** Nearly all public benchmark datasets (Section 5–6) are model-scale (towing tank, Froude-scaled) or model-scale CFD. ML surrogates trained on this data inherit the same Reynolds-number scale-effect uncertainty that ITTC's empirical extrapolation procedures exist to manage. Full-scale CFD is emerging as an alternative ground truth but is far more expensive to generate at the volume needed for ML training, so full-scale-validated ML surrogates remain rare.
2. **Multi-physics coupling (hydro–structural–propulsion).** Most ML work surveyed here is single-physics: resistance/seakeeping OR structural fatigue OR propulsion, rarely jointly. Genuinely coupled hydro-structural-propulsion ML (e.g., a surrogate that jointly predicts hull loads, structural response, and propulsion efficiency under a given sea state and operating condition) is largely absent from the public literature — most "digital twin" commercial products (Section 3) claim this integration but publish little methodological detail, so the actual state of the art in coupled multi-physics ML is difficult to assess independently.
3. **Data scarcity for extreme sea states.** Both ERA5 (documented to underestimate Hs at the upper end) and towing-tank/CFD datasets (built around design/operational conditions, not survival conditions) underrepresent extreme events. This is a significant problem for any ML model intended to support safety-critical predictions (capsize risk, extreme mooring loads, slamming) precisely in the regime where the consequences of model failure are highest.
4. **Generalization across hull forms.** Most surrogate models (Section 2.2, 2.6) are trained and validated on perturbations around a *single* parent hull form (e.g., variations of KCS) or a small closed set of benchmark hulls. Cross-hull-family generalization (training on KCS-family data, predicting accurately on a fundamentally different hull type) is not well demonstrated in the surveyed literature; geometric deep learning approaches operating directly on mesh/point-cloud representations (e.g., ShipNet) are a promising but still early direction for improving this.
5. **GNNs for mooring/offshore topology remain underexplored.** As noted in Section 2.4, the mooring/offshore-structure ML literature is dominated by feedforward/sequence ANNs rather than graph-native architectures that would naturally represent multi-line mooring topology or jacket/truss connectivity — despite GNNs being a natural fit and already used in adjacent structural-engineering domains.
6. **Real-time operational optimization under uncertainty.** Voyage optimization products (Nautilus Labs and similar) operate on relatively coarse (daily) update cycles combining weather forecasts with per-vessel performance models; genuinely real-time, uncertainty-aware optimization (jointly propagating forecast uncertainty, model uncertainty, and operational constraints) is more discussed as an aspiration in review papers than demonstrated at production scale in public sources.
7. **Offshore platform geometry lacks a shared benchmark set.** Unlike ship hulls (KCS/JBC/DTMB 5415/KVLCC), there is no widely-adopted small set of public reference offshore-platform geometries analogous to the Gothenburg Workshop hulls, making offshore-structure ML harder to benchmark and compare across papers (Section 5).
8. **Class-society treatment of ML (as distinct from conventional simulation) design tools is not well documented publicly.** As flagged in Section 7, specific rule language on how DNV/ABS/LR qualify ML-based design or monitoring tools for certification purposes was not located in this research pass; this is both a genuine gap in public documentation and a concrete opportunity for a reference repo to add value by systematically collecting and tracking it directly from primary rule sources.
9. **Standards for autonomous-ship AI are brand new and untested.** The IMO MASS Code took effect only in July 2026 and is voluntary; there is essentially no track record yet of how physics-AI-based autonomy/collision-avoidance systems will be certified against it, making this an active and fast-moving area to track rather than a settled reference.

---

## 9. Roadmap: Building the Reference Hub

A phased plan for this repository to become a genuinely authoritative, actively-used open-source reference for "Physics AI in Maritime" over roughly 1–3 years.

### Phase 1 (Months 0–3): Foundation and credibility
- Publish this landscape document with rigorous, checkable citations; explicitly mark unverified claims (as done above) and invite community correction via issues/PRs — credibility from day one matters more than completeness.
- Stand up a structured, machine-readable **bibliography** (BibTeX/CSL-JSON) of every paper cited here plus ongoing additions, tagged by sub-topic (resistance, seakeeping, wave forecasting, propulsion, autonomy, structures) so it's queryable, not just a prose list.
- Create a **benchmark geometry mirror/index**: canonical, clearly-licensed links (not unauthorized redistribution) to KCS, KVLCC2, DTMB 5415, and JBC geometry and ITTC EFD data sources, plus a short "how to use this benchmark" guide for newcomers.
- Set up basic repo infrastructure: contribution guidelines, a taxonomy/tagging scheme matching this document's section structure, and a lightweight review process for new entries (to prevent link-rot and unverifiable claims from accumulating).

### Phase 2 (Months 3–9): Depth and tooling
- Build a **curated, deduplicated dataset index** covering AIS sources (AISdb, AISHub, published research datasets), ocean/wave reanalysis (ERA5 access patterns, NDBC download tooling), and synthetic CFD-generation recipes (parametric hull deformation + open-source solver configs, e.g., Naval Hydro Pack/naoe-FOAM-SJTU starter cases for KCS/JBC) so newcomers can reproduce a baseline surrogate-training pipeline end-to-end without months of setup.
- Add a **"who's who"** directory: research groups (MIT, TU Delft, NTNU AMOS, Strathclyde NAOME, KRISO, SJTU, and others as verified), with links to lab pages and recent publication lists — kept current via periodic (e.g., quarterly) community-sourced review rather than one-time compilation.
- Add a **commercial landscape tracker** (the Section 3 table, expanded and kept current), clearly separating verified product claims from marketing claims, and explicitly noting where companies decline to publish methodology (common in this space) so readers calibrate confidence appropriately.
- Begin an **open benchmark/leaderboard experiment**: e.g., a standardized KCS resistance-prediction surrogate task with a fixed train/test split and clear metrics, allowing different groups' methods (PINN, FNO, GNN, classical Kriging) to be compared on genuinely equal footing — something the field currently lacks (each paper uses its own split and CFD settings, making cross-paper comparison unreliable).

### Phase 3 (Months 9–18): Community and standards engagement
- Actively track and summarize **ITTC procedure updates** and **class-society rule changes** relevant to simulation/AI-based design and certification (Section 7/8 gap #8) — this is a concrete, currently-unfilled niche the repo can own.
- Track **IMO MASS Code** implementation as it moves from voluntary (2026) toward the mandatory SOLAS track (targeted ~2030 adoption, ~2032 entry into force), summarizing what each milestone means for physics-AI-based autonomy systems.
- Establish light community touchpoints: a recurring (e.g., quarterly) roundup of new papers/products, structured for skimmability; consider aligning release cadence with major venues (OMAE, SNAME Maritime Convention) so the roundup is timely around when new work is presented.
- Solicit direct contributions from named research groups/companies identified in Sections 1–3 — invite corrections, updates, and (where appropriate) guest-curated sub-sections in their area of expertise, which both improves accuracy and builds the credibility/adoption needed for the repo to become a genuine reference point rather than a static document.

### Phase 4 (Months 18–36): Becoming the default reference
- Pursue explicit recognition/backlinks from ITTC, SNAME, or OMAE-adjacent community pages if genuinely earned through sustained accuracy and usefulness — do not claim affiliation prematurely.
- Expand into **reproducible tutorial content**: worked, runnable examples (e.g., a Jupyter/Colab notebook reproducing an FNO wave-forecast baseline on public ERA5/NDBC data, or a surrogate-optimization loop on KCS using an open-source solver) — tutorials are what convert a reference document into a hub people actually return to and cite.
- Maintain the open challenges list (Section 8) as a living, dated "state of the field" tracker — re-assess yearly whether each listed gap has closed, narrowed, or remains open, with citations to whatever progress has been made; this kind of longitudinal tracking is rare in the field and would be genuinely novel value-add.
- Consider light governance (a small maintainer group spanning at least one academic and one industry perspective) once contribution volume justifies it, to keep the "reference hub" positioning credible and non-partisan across the commercial landscape it documents.

---

## 10. Where this reference meets the PINNeAPPle codebase

*Added November 2026.* Everything above describes the field. This section is the other half: the places inside the PINNeAPPle organisation where each topic has something runnable, so a claim here can be checked against a case rather than taken on trust. Links are to live issues and finished experiments, not aspirations.

### Use cases and open work in [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle)

| Topic from this document | Where it lives |
|---|---|
| Section 4 — hydrostatics, resistance and pressure hulls | [#1203](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1203) standard workflow for submarines and underwater vehicles: hydrostatics, ITTC-57 resistance, pressure hull, depth control |
| Section 4 — marine control loops | [#1053](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1053) control cases 4/6: marine |
| Section 5 — open hull geometry | [#668](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/668) geometry library fidelity where `info()` flags estimates, including the KCS container-ship offsets |

### Benchmarks

There is no maritime experiment in `PINNeAPPle-Benchmark` yet, and this document does not pretend otherwise. The nearest reusable pieces are the resistance and hydrostatics workflow above and the general equal-budget comparison in [`09_equal_budget_pinn_fno_deeponet`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet), which is not a hull study. Citing one as a maritime benchmark would be the kind of overclaim Section 8 warns about.

### Related repositories

- [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle) — the library itself: PINNs, neural operators, solvers, geometry, validation.
- [`PINNeAPPle-CFD`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD) — the end-to-end CFD workflow behind the resistance cases.

---

## 11. Contributing

This document is curated, and corrections are the most valuable contribution it receives.

1. Open an issue or a pull request with the specific addition, correction or removal.
2. **Cite a real, checkable source** — a paper, an official repo, a vendor page. Anything not verifiable is marked as such rather than stated as fact.
3. Keep entries factual and neutral; no marketing language for commercial products, including PINNeAPPle's own.
4. When adding a dataset, tool or solver, say what it is, who maintains it and where it lives.
5. Flag what has gone stale — funding rounds, acquisitions and renamed products move quickly in this field.

The full conventions are in [CONTRIBUTING.md](../CONTRIBUTING.md). If you are adding something that has a runnable counterpart in the PINNeAPPle organisation, add a row to Section 10 as well so the claim stays checkable.

---

## 12. Sources

**Academic surveys and landmark papers**
- Weymouth & Yue, "Physics-Based Learning Models for Ship Hydrodynamics" — https://arxiv.org/abs/1401.3816
- Panda, "Machine Learning for Naval Architecture, Ocean and Marine Engineering" — https://arxiv.org/abs/2109.05574
- "A Scoping Review on Simulation-based Design Optimization in Marine Engineering: Trends, Best Practices, and Gaps" — https://arxiv.org/abs/2404.18654
- "ShipNet: A Geometric Deep Learning Surrogate for Real-Time Ship Hydrodynamics" — https://arxiv.org/abs/2606.15356
- "Predicting Ship Responses in Different Seaways using a Generalizable Force Correcting Machine Learning Method" — https://arxiv.org/abs/2405.08033
- "Physics-guided Neural Network-based Shaft Power Prediction for Vessels" — https://arxiv.org/abs/2512.20348
- "Physics informed neural networks based identification modelling of ship maneuvering motion..." (Eng. Appl. CFD, 2025) — https://www.tandfonline.com/doi/full/10.1080/19942060.2025.2566860
- "A CFD-informed machine learning framework for generalized ship resistance prediction under wind and wave conditions" — https://www.sciencedirect.com/science/article/abs/pii/S0029801826005937
- "Bridging ocean wave physics and deep learning: Physics-informed neural operators for nonlinear wavefield reconstruction in real-time" — https://arxiv.org/abs/2508.03315
- "Principled Fourier Neural Operators for High-Resolution Regional Ocean Modeling" (Jahanmard et al.) — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH001131
- "Applications of the Fourier neural operator in a regional ocean modeling and prediction" (Frontiers in Marine Science, 2024) — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1383997/full
- "Multiscale Neural PDE Surrogates for Prediction and Downscaling: Application to Ocean Currents" — https://arxiv.org/abs/2507.18067
- "DVL-DeepONet: A Physics-Guided Operator Learning for Resilient Underwater Navigation" — https://arxiv.org/abs/2606.23502
- "Deep physics for deep seas: Estimating underwater vehicle dynamics with PINNs" (Ocean Engineering) — https://www.sciencedirect.com/science/article/pii/S0029801826017531
- "Research on model predictive control of autonomous underwater vehicle based on physics informed neural network modeling" — https://www.sciencedirect.com/science/article/abs/pii/S002980182401182X
- "Physics-informed compact world models for predictive control in AUV terminal docking under communication constraints" — https://link.springer.com/article/10.1007/s41315-026-00530-1
- "AI-Aided Advancements in Autonomous Underwater Vehicle Navigation" — https://arxiv.org/abs/2605.04672
- "AI-Driven Performance-to-Design Generation and Optimization of Marine Propellers" — https://arxiv.org/abs/2604.22224
- "Fusion of ML with numerical simulation for optimized propeller design" — https://arxiv.org/abs/2302.14740
- "A Novel Reinforcement Learning Collision Avoidance Algorithm for USVs Based on Maneuvering Characteristics and COLREGs" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8954226/
- "COLREGs-compliant multiship collision avoidance based on deep reinforcement learning" — https://www.sciencedirect.com/science/article/abs/pii/S0029801819305840
- "Trust, Geometry, and Rules: A Credibility-Aware Reinforcement Learning Framework for Safe USV Navigation under Uncertainty" — https://arxiv.org/abs/2605.26974
- "Multiple Ships Cooperative Navigation and Collision Avoidance using Multi-agent Reinforcement Learning with Communication" — https://arxiv.org/abs/2410.21290
- "Multi-scale Dynamic Wake Modeling and Prediction of Floating Offshore Wind Turbines via Physics-Informed Neural Networks and Fourier Neural Operators" — https://arxiv.org/abs/2604.23937
- "Real-Time Digital Twin for Structural Health Monitoring of Floating Offshore Wind Turbines" — https://doi.org/10.3390/jmse13101953
- "Artificial Intelligence in Floating Offshore Wind Turbines: A Critical Review" — https://www.mdpi.com/1996-1073/18/22/5937
- "AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review" — https://arxiv.org/abs/2505.07374
- "Maritime Tracking Data Analysis and Integration with AISdb" — https://arxiv.org/abs/2407.08082
- "Dataset of ocean vessel traffic in the North Sea" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10663730/
- "Ocean Wave Forecasting with Deep Learning as Alternative to Conventional Models" — https://arxiv.org/abs/2406.03848
- "Global evaluation of wave data reanalysis: Comparison of the ERA5 dataset to buoy observations" — https://www.sciencedirect.com/science/article/pii/S0141118725000781
- "SCAWaveNet: A Spatial-Channel Attention-Based Network for Global Significant Wave Height Retrieval" — https://arxiv.org/abs/2507.00701
- "Data-Driven Reconstruction of Significant Wave Heights from Sparse Observations" — https://arxiv.org/abs/2509.19384
- "Scale effects and full-scale ship hydrodynamics: A review" — https://www.sciencedirect.com/science/article/abs/pii/S0029801821017741
- "Full-scale CFD simulations for the determination of ship resistance as a rational, alternative method to towing tank experiments" — https://www.sciencedirect.com/science/article/abs/pii/S0029801819305839

**Institutions**
- TU Delft — "Physics-based learning models for ship hydrodynamics" — https://research.tudelft.nl/en/publications/physics-based-learning-models-for-ship-hydrodynamics-2
- NTNU AMOS (Centre for Autonomous Marine Operations and Systems) — https://www.ntnu.edu/amos
- University of Strathclyde NAOME MSc Autonomous Marine Vehicles — https://www.strath.ac.uk/courses/postgraduatetaught/marineengineeringwithspecialisationinautonomousmarinevehicles/

**Numerical methods / solvers**
- SWAN model documentation — https://swanmodel.sourceforge.io/
- Naval Hydro Pack (foam-extend) — https://openfoam-extend.sourceforge.net/OpenFOAM_Workshops/OFW11_2016_Guimaraes/special.html
- naoe-FOAM-SJTU solver — https://link.springer.com/article/10.1007/s42241-019-0020-6
- ITTC Recommended Procedures and Guidelines (benchmark database, V&V) — https://ittc.info/media/2115/75-03-02-01.pdf , https://www.ittc.info/media/11956/75-03-02-02.pdf , https://ittc.info/media/6097/sc-cfd.pdf

**Geometry / benchmark hulls**
- KRISO Container Ship (KCS) — NMRI Gothenburg 2000 archive — https://www.nmri.go.jp/study/research_organization/fluid_performance/cfd/cfdws05/gothenburg2000/KCS/container.html
- "Benchmarking of Computational Fluid Dynamics for Ship Flows: The Gothenburg 2000 Workshop" — https://www.researchgate.net/publication/233502941

**Standards / regulation**
- IMO — Autonomous Shipping (MASS) overview — https://www.imo.org/en/ourwork/safety/pages/autonomous-shipping-(mass).aspx
- IMO — "IMO adopts first global Code for autonomous ships" — https://imo-newsroom.prgloo.com/news/imo-adopts-first-global-code-for-autonomous-ships
- DNV — "IMO MSC 111: New MASS Code adopted" — https://www.dnv.com/news/2026/imo-mcs-111-new-mass-code-adopted/
- DNV — RuleAgent coverage — https://www.rivieramm.com/news-content-hub/ai-enables-crews-managers-to-access-dnv-class-rules-faster-88280

**Companies / products**
- DNV digital twinning — https://www.dnv.com/maritime/dutch-naval-design/digital-twinning/
- Kongsberg Digital — Kognitwin rename announcement — https://kongsbergdigital.com/news/kongsberg-digitals-digital-twin-solutions-renamed-to-kognitwin
- ABB Ability Marine Pilot Vision — https://new.abb.com/news/detail/51627/abb-ability-marine-pilot-vision-looks-beyond-human-vision-for-ship-automation
- ABB Ability Marine Pilot Control (Suomenlinna II trial) — https://new.abb.com/news/detail/24659/paving-the-way-towards-autonomous-shipping-abb-ability-marine-pilot-control
- NAPA — https://www.napa.fi/
- Windward — https://windward.ai/
- Orca AI — https://www.orca-ai.io/about/ ; funding — https://www.calcalistech.com/ctechnews/article/bkotyuvgle
- Nautilus Labs — https://nautiluslabs.com/solutions/voyage-optimizer/
- ShipIn Systems — https://www.shipin.ai/
- Sofar Ocean — https://jobs.schmidtmarine.org/companies/sofar-ocean-technologies/jobs/38363408-maritime-data-analyst
- Siemens / Compute Maritime NeuralShipper–STAR-CCM+ integration — https://resources.sw.siemens.com/en-US/e-book-marine-cfd-software

---

*Compiled September 2026. This document is intended as a seed for an actively maintained open-source repository — not a final or exhaustive account. Flagged uncertainties (Compute Maritime/NeuralShipper detail depth, "Prospection"/"Deep Vision" companies, specific class-society ML-qualification rule text, and the maturity of GNN methods for mooring analysis) should be prioritized for follow-up verification by contributors.*
