---
id: ASSET-WT
title: Process stages of the water-treatment testbed
tags: [asset, process stage, raw water, chemical dosing, ultrafiltration, dechlorination, reverse osmosis, backwash, control network]
references:
  - SWaT testbed description, iTrust, Singapore University of Technology and Design
---
## Stage P1 raw water intake
The raw-water tank receives water through a motorised inlet valve and feeds the next stage with a transfer pump. Its level transmitter and flow transmitters drive the inlet valve and pump interlocks. Manipulation can overflow or drain the tank and damage pumps that run dry.

## Stage P2 chemical dosing
Dosing pumps add chemicals controlled by pH, conductivity and redox analyzers. This stage is safety-relevant because incorrect dosing directly affects water quality.

## Stage P3 ultrafiltration
Ultrafiltration membranes remove particles; differential pressure indicates fouling and triggers backwash. Manipulation can damage membranes or pass untreated water downstream.

## Stages P4 to P6
Dechlorination, reverse osmosis and backwash complete the treatment. Reverse-osmosis membranes are expensive and sensitive to chlorine and pressure excursions.

## Control network
The control network connects controllers, the HMI, the historian and engineering workstations. Its traffic volume, destinations and write commands are monitored as security telemetry.
