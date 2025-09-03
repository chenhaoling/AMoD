# AMoD
---
# **SUMO ride-hailling parking model simulation**
This repository implements a comprehensive traffic simulation system focusing on roadside parking behavior and emission analysis in Beijing.
 The system integrates SUMO with advanced optimization algorithms for fleet management and ride-hailing dispatch.

---
- **project**
  - **config**
  - **paths**
  - **simulation_setup**
  - **od_demand_generator**
  - **fleet_optimizer**
  - **interfaces**
  - **dispatch_manager**
  - **ridehail_coordinator**
  - **parking_manager**
  - **detector**
  - **emissions**



---


   # Main Modules Overview

## config
Defines simulation parameters including area boundaries, vehicle densities, parking ratios, and emission factors. Supports both local and city-wide simulation modes.

## paths
Manages file paths, detects SUMO installation locations, and organizes output directories. Ensures cross-platform compatibility for SUMO tools.

## simulation_setup
After downloading the road network data based on OSM, convert it into a SUMO network and generate vehicle routes. Use randomTrips.py to generate initial trip data.

## od_demand_generator
Implements Origin-Destination (OD) matrix-based demand generation, incorporating Beijing-specific travel patterns. Creates zone-based traffic analysis areas and generates realistic trip distributions.

## fleet_optimizer
Optimizes vehicle shareability based on MIT research. Constructs shareability networks and solves minimum path cover problems to determine the optimal fleet size.

## interfaces
Provides modular interfaces for demand generation, fleet optimization, dispatch optimization, and parking decisions. Implements algorithms based on the "Nature" paper for fleet optimization and real-time dispatching.

## dispatch_manager
A comprehensive ride-hailing dispatch system with driver-order matching algorithms. Calculates distances using the OSMNX formula, manages driver preferences, and maintains rejection lists.

## ridehail_coordinator
Coordinates order generation, vehicle dispatch, and parking decisions post-service. Integrates with the main simulation loop for real-time ride-hailing operations.

## parking_manager
Estimates roadside parking density based on road types and urban characteristics. Implements parking spot distribution algorithms tailored for different road categories.

## detector
Performs real-time detection and management of parking events during simulation. Handles parking position validation, traffic impact analysis, and emission calculations.

## emissions
Calculates vehicle emissions based on the HBEFA4 model, supporting gasoline, diesel, and electric vehicles. Implements driving mode detection and CO₂-equivalent calculations.

## traci_manager
Manages SUMO TraCI connections with automatic retry mechanisms and error handling.



