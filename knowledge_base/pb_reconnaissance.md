---
id: PB-RECON
title: Reconnaissance and discovery as an early warning of an attack
tags: [reconnaissance, scanning, discovery, precursor, early warning, predicted attack, hosts, enumeration]
references:
  - MITRE ATT&CK for ICS T0846 Remote System Discovery
  - MITRE ATT&CK for ICS T0840 Network Connection Enumeration
  - MITRE ATT&CK for ICS T0888 Remote System Information Discovery
  - MITRE ATT&CK Enterprise T1046 Network Service Discovery
---
## Indicators
A host contacts many more destinations than usual on the control network, probes industrial protocol ports, or causes failed authentications. Discovery activity often precedes manipulation by minutes to hours, so it is the main input for predicting an attack before physical effects appear. Legitimate asset-inventory scans can look similar and must be ruled out. These behaviours map to T0846, T0840 and T0888.

## Investigation steps
1. Identify the scanning source host and check whether it is an approved asset-inventory or vulnerability-management system with a scheduled scan.
2. List the destinations and protocol ports that were probed and which controllers responded.
3. Correlate the scanning window with authentication failures and remote-access sessions.
4. Increase monitoring of the controllers that were probed and ask operations to report unusual process behaviour during the predicted horizon.

## Containment requiring approval
1. Restrict the scanning host at the zone boundary if it is not an approved scanner, after confirming it carries no legitimate control traffic.
2. Notify the operations shift lead that an attack may follow within the predicted horizon so that manual fallback procedures are ready.
