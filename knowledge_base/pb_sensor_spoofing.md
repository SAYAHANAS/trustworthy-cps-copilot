---
id: PB-SPOOF
title: Sensor spoofing and false data injection on process measurements
tags: [sensor, spoofing, false data injection, level, flow, measurement integrity, replay, reporting message]
references:
  - MITRE ATT&CK for ICS T0856 Spoof Reporting Message
  - MITRE ATT&CK for ICS T0832 Manipulation of View
  - MITRE ATT&CK for ICS T0831 Manipulation of Control
  - NIST SP 800-82 Guide to Operational Technology Security
---
## Indicators
A transmitter such as a tank level or flow sensor reports values that are physically inconsistent with related measurements. Typical signs are a level reading that stays flat while inflow continues, replayed values with natural-looking noise, a sudden jump when the spoofing stops, and controller actions that follow the false reading instead of the real process state. Spoofed reporting messages map to T0856 and the resulting loss of operator visibility to T0832.

## Investigation steps
1. Compare the suspect sensor with physically related measurements using a mass or energy balance, for example inflow and outflow against the change in tank level.
2. Ask operations to read the local field gauge or a redundant transmitter and compare it with the value shown in the HMI and historian.
3. Review controller and historian logs for the period before the deviation for write commands or configuration changes affecting the sensor tag.
4. Capture control-network traffic between the field device, controller and HMI to check whether reporting messages are replayed or altered in transit.
5. Check whether automatic control loops acted on the suspect value and estimate the true process state they produced.

## Containment requiring approval
1. With the shift supervisor, move the affected control loop to manual or local control following the plant standard operating procedure.
2. Isolate the suspected manipulation path at the zone boundary only after operations confirms the process can run safely without it.
3. Preserve controller programs, HMI project files and network captures as forensic evidence before any restart or reload.

## Recovery
1. Restore the sensor signal path and verify readings against an independent reference before returning loops to automatic control.
2. Re-baseline alarm limits that may have been modified during the incident.
